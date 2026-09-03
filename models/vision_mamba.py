import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from einops import rearrange, repeat
except ImportError:
    rearrange = None
    repeat = None
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    selective_scan_fn = None

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x


def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """
    Pure PyTorch implementation of selective scan for portability and verification.
    u: (B, L, D_inner)
    delta: (B, L, D_inner)
    A: (D_inner, D_state)
    B: (B, L, D_state)
    C: (B, L, D_state)
    D: (D_inner,)
    """
    if delta_bias is not None:
        delta = delta + delta_bias[None, None, :]
    if delta_softplus:
        delta = F.softplus(delta)

    batch, seqlen, dim = u.shape
    dstate = A.shape[-1]

    # Scan step-by-step over sequence dimension without allocating (B, L, dim, dstate) upfront
    A_unsq = A.unsqueeze(0)  # (1, dim, dstate)
    h = torch.zeros(batch, dim, dstate, device=u.device, dtype=u.dtype)
    ys = []

    for i in range(seqlen):
        delta_i = delta[:, i]  # (B, dim)
        u_i = u[:, i]          # (B, dim)
        B_i = B[:, i]          # (B, dstate)
        C_i = C[:, i]          # (B, dstate)

        dA_i = torch.exp(delta_i.unsqueeze(-1) * A_unsq)  # (B, dim, dstate)
        dBu_i = (delta_i * u_i).unsqueeze(-1) * B_i.unsqueeze(1)  # (B, dim, dstate)

        h = dA_i * h + dBu_i
        y_i = torch.einsum('bdn,bn->bd', h, C_i)  # (B, dim)
        ys.append(y_i)

    y = torch.stack(ys, dim=1)  # (B, L, dim)

    if D is not None:
        y = y + u * D[None, None, :]
    return y

def selective_scan_cuda_compat(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """
    CUDA-backed selective scan compatible with selective_scan_ref.

    If the optional mamba_ssm extension is unavailable, fall back to the
    portable reference implementation. This is required for a clean B1 baseline
    in environments that do not ship the extension, without changing the model
    architecture or training objective.
    """
    if selective_scan_fn is None:
        return selective_scan_ref(
            u=u,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=D,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
        )

    # Convert from this project's [B, L, D] layout to
    # mamba_ssm's [B, D, L] layout.
    u_ssm = u.transpose(1, 2).contiguous()
    delta_ssm = delta.transpose(1, 2).contiguous()

    # Convert B/C from [B, L, N] to [B, N, L].
    B_ssm = B.transpose(1, 2).contiguous()
    C_ssm = C.transpose(1, 2).contiguous()

    y = selective_scan_fn(
        u_ssm,
        delta_ssm,
        A,
        B_ssm,
        C_ssm,
        D=D,
        z=None,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=False,
    )

    # Convert back to [B, L, D].
    return y.transpose(1, 2).contiguous()


class BiMamba(nn.Module):
    """
    Bidirectional Mamba block supporting bimamba_type='v2'.
    Matches the official hustvl/Vim checkpoint structure.
    """
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        bimamba_type: str = "v2",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.bimamba_type = bimamba_type

        # Linear in-projection for forward and gate
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        # 1D Depthwise Convolutions
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        if self.bimamba_type == "v2":
            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
            )

        # SSM Projection layers
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        if self.bimamba_type == "v2":
            self.x_proj_b = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        if self.bimamba_type == "v2":
            if dt_init == "constant":
                nn.init.constant_(self.dt_proj_b.weight, dt_init_std)
            elif dt_init == "random":
                nn.init.uniform_(self.dt_proj_b.weight, -dt_init_std, dt_init_std)
            with torch.no_grad():
                self.dt_proj_b.bias.copy_(inv_dt)
            self.dt_proj_b.bias._no_reinit = True

        # S4D real initialization for A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        if self.bimamba_type == "v2":
            A_b = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_b_log = nn.Parameter(torch.log(A_b))
            self.A_b_log._no_weight_decay = True

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        if self.bimamba_type == "v2":
            self.D_b = nn.Parameter(torch.ones(self.d_inner))
            self.D_b._no_weight_decay = True

        # Out projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B, L, D)
        """
        B, L, _ = u.shape

        # In-projection to get hidden and gate
        xz = self.in_proj(u)  # (B, L, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 1. Forward scan branch
        x_fwd = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)  # (B, L, d_inner)
        x_fwd = F.silu(x_fwd)

        x_dbl = self.x_proj(x_fwd)  # (B, L, dt_rank + 2 * d_state)
        dt, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt)  # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        y_fwd = selective_scan_cuda_compat(
            x_fwd, dt, A, B_proj, C_proj, self.D.float(), delta_softplus=True
        )

        # 2. Backward scan branch (BiMamba v2)
        if self.bimamba_type == "v2":
            x_bwd = torch.flip(x, dims=[1])
            x_bwd = self.conv1d_b(x_bwd.transpose(1, 2))[:, :, :L].transpose(1, 2)
            x_bwd = F.silu(x_bwd)

            x_dbl_b = self.x_proj_b(x_bwd)
            dt_b, B_proj_b, C_proj_b = torch.split(
                x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt_b = self.dt_proj_b(dt_b)

            A_b = -torch.exp(self.A_b_log.float())
            y_bwd = selective_scan_cuda_compat(
                x_bwd, dt_b, A_b, B_proj_b, C_proj_b, self.D_b.float(), delta_softplus=True
            )
            y_bwd = torch.flip(y_bwd, dims=[1])
            # Official Vim BiMamba-v2 averages the forward and reverse scans
            # before the shared gate and output projection.
            y = (y_fwd + y_bwd) / 2
        else:
            y = y_fwd

        # Gated output
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


class Block(nn.Module):
    """
    Vim Block wrapping mixer (BiMamba) with RMSNorm and pre-norm residual.
    """
    def __init__(
        self,
        dim: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        drop_path: float = 0.0,
        bimamba_type: str = "v2",
        norm_layer=RMSNorm,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mixer = BiMamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            bimamba_type=bimamba_type,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Follows standard pre-norm residual logic.
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.norm(hidden_states)

        hidden_states = self.drop_path(self.mixer(hidden_states))
        return hidden_states, residual


class VisionMamba(nn.Module):
    """
    Vision Mamba (Vim-Base) implementation matching official hustvl/Vim architecture.
    """
    def __init__(
        self,
        img_size: int = 448,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 198,
        embed_dim: int = 768,
        depth: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        use_middle_cls_token: bool = True,
        bimamba_type: str = "v2",
        norm_layer=RMSNorm,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        self.use_middle_cls_token = use_middle_cls_token

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=None,
        )
        num_patches = self.patch_embed.num_patches

        # Middle CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position Embedding (196 spatial + 1 cls for 224, 784 spatial + 1 cls for 448)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # 24 Mamba Blocks
        self.layers = nn.ModuleList([
            Block(
                dim=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                drop_path=dpr[i],
                bimamba_type=bimamba_type,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        # Final Normalization and Classification Head
        self.norm_f = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Checkpoints for intermediate feature extraction (blocks 6, 12, 18, 24)
        self.tapped_layers = [6, 12, 18, 24]

        # Foreground-Background Feature Distillation (WeaklySelector)
        self.use_selection = True
        self.num_selects = {
            'layer1': 256,
            'layer2': 128,
            'layer3': 64,
            'layer4': 32,
        }
        feat_dims = {
            'layer1': embed_dim,
            'layer2': embed_dim,
            'layer3': embed_dim,
            'layer4': embed_dim,
        }
        self.selector = WeaklySelector(feat_dims, num_classes, self.num_selects)

        # Initialize weights
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            feats: Final 1D normalized class token feature [B, 768]
            feat_dict: Dictionary containing 4 intermediate spatial patch tensors [B, 784, 768]
                       from blocks 6, 12, 18, 24 (excluding the middle CLS token).
        """
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # [B, 784, 768]
        M = x.shape[1]

        # Insert CLS token at middle index (M // 2 = 392 for 784 patches)
        if self.use_middle_cls_token:
            token_position = M // 2
            cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
            x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)  # [B, 785, 768]
        else:
            token_position = 0
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Forward through Mamba blocks & tap intermediate states
        residual = None
        hidden_states = x
        feat_dict = {}

        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(hidden_states, residual)
            block_idx = i + 1

            if block_idx in self.tapped_layers:
                stage_name = f"layer{len(feat_dict) + 1}"
                # Compute authentic hidden state by adding residual
                hidden = hidden_states if residual is None else (hidden_states + residual)
                # Strip out middle CLS token to retain only spatial patch tokens [B, 784, 768]
                if self.use_middle_cls_token:
                    patch_tokens = torch.cat(
                        (hidden[:, :token_position, :], hidden[:, token_position + 1 :, :]),
                        dim=1,
                    )
                else:
                    patch_tokens = hidden[:, 1:, :]
                feat_dict[stage_name] = patch_tokens

        # Final residual addition
        hidden_states = hidden_states if residual is None else (hidden_states + residual)

        # Extract normalized CLS token feature
        if self.use_middle_cls_token:
            cls_out = hidden_states[:, token_position, :]
        else:
            cls_out = hidden_states[:, 0, :]

        feats = self.norm_f(cls_out)  # [B, 768]
        return feats, feat_dict

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        Returns:
            outputs: Classification logits [B, num_classes]
            feats: Penultimate 1D feature [B, 768]
            logits_dict: Background token logits {layer1: [B, 528, 198], ...}
        """
        feats, feat_dict = self.forward_features(x)
        outputs = self.head(feats)
        logits_dict = {}
        if self.use_selection:
            logits_dict = self.selector(feat_dict)
        return outputs, feats, logits_dict

    @staticmethod
    def _resize_middle_cls_pos_embed(
        checkpoint_pos_embed: torch.Tensor,
        target_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Resize an official Vim middle-CLS absolute position embedding.

        Vim places its class token in the middle of the patch sequence.  The
        class-token position must therefore be removed before treating the
        remaining entries as a 2-D patch grid, then inserted at the middle of
        the resized grid.
        """
        if checkpoint_pos_embed.ndim != 3 or target_pos_embed.ndim != 3:
            raise ValueError("Vim positional embeddings must have shape [1, tokens, channels].")
        if checkpoint_pos_embed.shape[0] != 1 or target_pos_embed.shape[0] != 1:
            raise ValueError("Only a single shared Vim positional embedding is supported.")
        if checkpoint_pos_embed.shape[-1] != target_pos_embed.shape[-1]:
            raise ValueError(
                "Checkpoint/model position embedding dimensions differ: "
                f"{checkpoint_pos_embed.shape[-1]} vs {target_pos_embed.shape[-1]}."
            )

        source_patch_tokens = checkpoint_pos_embed.shape[1] - 1
        target_patch_tokens = target_pos_embed.shape[1] - 1
        source_size = int(math.isqrt(source_patch_tokens))
        target_size = int(math.isqrt(target_patch_tokens))
        if source_size * source_size != source_patch_tokens or target_size * target_size != target_patch_tokens:
            raise ValueError("Vim positional embeddings must contain square patch grids plus one CLS token.")

        source_cls_index = source_patch_tokens // 2
        target_cls_index = target_patch_tokens // 2
        source_cls = checkpoint_pos_embed[:, source_cls_index:source_cls_index + 1]
        source_spatial = torch.cat(
            (checkpoint_pos_embed[:, :source_cls_index], checkpoint_pos_embed[:, source_cls_index + 1:]), dim=1
        )
        source_spatial = source_spatial.reshape(1, source_size, source_size, -1).permute(0, 3, 1, 2)
        resized_spatial = F.interpolate(
            source_spatial.float(), size=(target_size, target_size), mode="bicubic", align_corners=False
        ).to(dtype=checkpoint_pos_embed.dtype)
        resized_spatial = resized_spatial.permute(0, 2, 3, 1).reshape(1, target_patch_tokens, -1)
        return torch.cat(
            (resized_spatial[:, :target_cls_index], source_cls, resized_spatial[:, target_cls_index:]), dim=1
        )

    @staticmethod
    def _checkpoint_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
        """Extract and normalize a model state dict from an official Vim checkpoint."""
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
        if not isinstance(state_dict, Mapping):
            raise TypeError("Checkpoint does not contain a model state dictionary.")
        normalized = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            normalized[key.removeprefix("module.")] = value
        if not normalized:
            raise ValueError("Checkpoint contains no tensor parameters.")
        return normalized

    @staticmethod
    def _validate_checkpoint_candidate(
        candidate_state: Mapping[str, torch.Tensor],
        model_state: Mapping[str, torch.Tensor],
    ) -> List[str]:
        """Validate a candidate state dict without modifying model parameters.

        FDCL-DA deliberately initializes its classification and selector heads,
        so those are the only keys allowed to be absent from the official Vim
        checkpoint.
        """
        permitted_missing = sorted(
            key for key in model_state if key.startswith("head.") or key.startswith("selector.")
        )
        candidate_keys = set(candidate_state)
        model_keys = set(model_state)
        missing_keys = sorted(model_keys - candidate_keys - set(permitted_missing))
        unexpected_keys = sorted(candidate_keys - model_keys)
        shape_mismatches = {
            key: (tuple(candidate_state[key].shape), tuple(model_state[key].shape))
            for key in sorted(candidate_keys & model_keys)
            if tuple(candidate_state[key].shape) != tuple(model_state[key].shape)
        }
        if missing_keys or unexpected_keys or shape_mismatches:
            details = []
            if missing_keys:
                details.append(f"missing={missing_keys}")
            if unexpected_keys:
                details.append(f"unexpected={unexpected_keys}")
            if shape_mismatches:
                mismatch_text = ", ".join(
                    f"{key}: {source} != {target}"
                    for key, (source, target) in shape_mismatches.items()
                )
                details.append(f"shape_mismatches={{ {mismatch_text} }}")
            raise RuntimeError("Official Vim-Base checkpoint validation failed: " + "; ".join(details))
        return permitted_missing

    def load_official_vim_base_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load official ``vim_b_midclstok_81p9acc.pth`` weights for 448px FDCL-DA.

        The ImageNet-1K head is deliberately dropped.  FDCL-DA's 198-way head
        and selector heads remain newly initialized.  Returns an auditable load
        report, and raises before mutating model parameters for architecture
        incompatibilities.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = self._checkpoint_state_dict(checkpoint)

        architecture_errors: List[str] = []
        expected_shapes = {
            "patch_embed.proj.weight": (self.embed_dim, 3, 16, 16),
            "cls_token": (1, 1, self.embed_dim),
            "pos_embed": (1, 197, self.embed_dim),
            "norm_f.weight": (self.embed_dim,),
        }
        for key, shape in expected_shapes.items():
            tensor = state_dict.get(key)
            if tensor is None:
                architecture_errors.append(f"missing required checkpoint key: {key}")
            elif tuple(tensor.shape) != shape:
                architecture_errors.append(f"{key}: checkpoint {tuple(tensor.shape)} != expected {shape}")
        layer_indices = {
            int(key.split(".")[1]) for key in state_dict
            if key.startswith("layers.") and len(key.split(".")) > 2 and key.split(".")[1].isdigit()
        }
        if layer_indices != set(range(24)):
            architecture_errors.append(f"checkpoint layers are {sorted(layer_indices)}, expected indices 0..23")
        for key in ("layers.0.mixer.conv1d_b.weight", "layers.0.mixer.x_proj_b.weight", "layers.0.mixer.dt_proj_b.weight"):
            if key not in state_dict:
                architecture_errors.append(f"missing BiMamba-v2 key: {key}")
        if not self.use_middle_cls_token:
            architecture_errors.append("model is not configured with a middle CLS token")
        if not isinstance(self.norm_f, RMSNorm):
            architecture_errors.append(f"model final norm is {type(self.norm_f).__name__}, expected RMSNorm")
        if architecture_errors:
            raise RuntimeError("Official Vim-Base architecture check failed: " + "; ".join(architecture_errors))

        state_dict["pos_embed"] = self._resize_middle_cls_pos_embed(state_dict["pos_embed"], self.pos_embed)
        checkpoint_head_keys = [key for key in ("head.weight", "head.bias") if key in state_dict]
        for key in checkpoint_head_keys:
            state_dict.pop(key)

        model_state = self.state_dict()
        permitted_missing = self._validate_checkpoint_candidate(state_dict, model_state)
        incompatible = self.load_state_dict(state_dict, strict=False)
        returned_missing = sorted(incompatible.missing_keys)
        returned_unexpected = sorted(incompatible.unexpected_keys)
        if returned_missing != permitted_missing or returned_unexpected:
            raise RuntimeError(
                "load_state_dict returned keys inconsistent with validated checkpoint: "
                f"missing={returned_missing}; unexpected={returned_unexpected}"
            )

        loaded_keys = sorted(key for key in state_dict if key in model_state)
        return {
            "loaded_parameter_keys": len(loaded_keys),
            "loaded_parameter_numel": sum(model_state[key].numel() for key in loaded_keys),
            "missing_keys": returned_missing,
            "unexpected_keys": returned_unexpected,
            "harmless_missing_keys": permitted_missing,
            "discarded_checkpoint_head_keys": checkpoint_head_keys,
            "source_pos_embed_shape": (1, 197, self.embed_dim),
            "target_pos_embed_shape": tuple(self.pos_embed.shape),
        }


class WeaklySelector(nn.Module):
    """
    Foreground-background token selection module.
    Projects intermediate patch tokens to class logits, identifies principal entity
    tokens with highest class prediction confidence, and extracts background tokens for distillation.
    """
    def __init__(self, inputs: Union[dict, Dict[str, Union[torch.Tensor, int]]], num_classes: int, num_select: dict):
        super(WeaklySelector, self).__init__()
        self.num_select = num_select
        self.num_classes = num_classes

        # Build classifier heads for each layer
        for name, feat in inputs.items():
            if isinstance(feat, int):
                in_size = feat
            elif isinstance(feat, torch.Tensor):
                fs_size = feat.size()
                if len(fs_size) == 3:
                    in_size = fs_size[2]
                elif len(fs_size) == 4:
                    in_size = fs_size[1]
                else:
                    raise ValueError(f"Unsupported feature tensor rank: {len(fs_size)}")
            else:
                raise TypeError(f"Unsupported input type for {name}: {type(feat)}")

            m = nn.Linear(in_size, num_classes)
            self.add_module("classifier_l_" + name, m)

    def forward(self, x: Dict[str, torch.Tensor], logits=None) -> Dict[str, torch.Tensor]:
        logits = {}
        logits_distillation = {}
        for name in x:
            feat = x[name]
            if len(feat.size()) == 4:
                B, C, H, W = feat.size()
                feat = feat.view(B, C, H * W).permute(0, 2, 1).contiguous()

            classifier = getattr(self, "classifier_l_" + name)
            logits[name] = classifier(feat).float()
            probs = torch.softmax(logits[name], dim=-1)
            sum_probs = torch.softmax(logits[name].mean(1), dim=-1)

            preds_0 = []
            num_select = self.num_select[name]
            for bi in range(logits[name].size(0)):
                _, max_ids = torch.max(sum_probs[bi], dim=-1)
                confs, ranks = torch.sort(probs[bi, :, max_ids], descending=True)
                preds_0.append(logits[name][bi][ranks[num_select:]].float())

            preds_0 = torch.stack(preds_0)
            logits_distillation[name] = preds_0.float()

        return logits_distillation
