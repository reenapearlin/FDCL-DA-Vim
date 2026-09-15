import torch
import torch.nn as nn
import mamba_ssm.ops.selective_scan_interface as ssi

# Route selective scan to pure PyTorch reference
ssi.selective_scan_fn = ssi.selective_scan_ref

from transformers import AutoModel

class MambaVisionFDCL(nn.Module):
    def __init__(self, num_classes=198):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            'nvidia/MambaVision-B-1K', 
            trust_remote_code=True
        )
        self.embed_dim = 1024
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward_features(self, x):
        pooler, features = self.backbone(x)
        feat_map = features[-1]
        return pooler, feat_map

    def forward(self, x):
        pooler, _ = self.forward_features(x)
        return self.head(pooler)
