import os
import sys
import logging
import argparse
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from models.mambavision_da_fd import MambaVisionDAFD

def setup_logger(log_file="training_da_fd_200.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class DualAugmentSoyDataset(Dataset):
    def __init__(self, anno_file, img_dir, std_transform=None, aux_transform=None):
        self.img_dir = img_dir
        self.std_transform = std_transform
        self.aux_transform = aux_transform
        self.samples = []
        with open(anno_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.samples.append((parts[0], int(parts[1])))
        min_label = min(s[1] for s in self.samples)
        if min_label == 1:
            self.samples = [(s[0], s[1] - 1) for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        img_sa = self.std_transform(image) if self.std_transform else image
        img_aa = self.aux_transform(image) if self.aux_transform else img_sa
        return img_sa, img_aa, label

def patch_shuffle(img_tensor, grid_size=4):
    B, C, H, W = img_tensor.shape
    p_h, p_w = H // grid_size, W // grid_size
    patches = img_tensor.unfold(2, p_h, p_h).unfold(3, p_w, p_w)
    patches = patches.contiguous().view(B, C, grid_size * grid_size, p_h, p_w)
    shuffled_batches = []
    for b in range(B):
        idx = torch.randperm(grid_size * grid_size)
        shuffled = patches[b, :, idx, :, :].view(C, grid_size, grid_size, p_h, p_w)
        shuffled = shuffled.permute(0, 1, 3, 2, 4).contiguous().view(C, H, W)
        shuffled_batches.append(shuffled)
    return torch.stack(shuffled_batches)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/home/subsea/reena/intern/VMambaFDCL/data/SoyAgeing-R1/R1')
    parser.add_argument('--start_epoch', type=int, default=121)
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--num_classes', type=int, default=198)
    parser.add_argument('--lambda_fd', type=float, default=1.25)
    parser.add_argument('--checkpoint', type=str, default='mambavision_da_fd_best.pth')
    args = parser.parse_args()

    logger = setup_logger("training_da_fd_200.log")
    logger.info(f"Resuming DA+FD to 200 epochs (Epoch {args.start_epoch} -> {args.total_epochs}) | Base Checkpoint: {args.checkpoint}")

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_sa_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])

    train_aa_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img_dir = os.path.join(args.base_dir, 'images')
    anno_dir = os.path.join(args.base_dir, 'anno')

    train_set = DualAugmentSoyDataset(
        os.path.join(anno_dir, 'train.txt'), 
        img_dir, 
        std_transform=train_sa_transform,
        aux_transform=train_aa_transform
    )
    val_set = DualAugmentSoyDataset(
        os.path.join(anno_dir, 'val.txt'), 
        img_dir, 
        std_transform=val_transform,
        aux_transform=val_transform
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = MambaVisionDAFD(num_classes=args.num_classes).to(device)

    if os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded existing best weights from {args.checkpoint}")
    else:
        logger.error(f"Checkpoint {args.checkpoint} not found!")
        return

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=(args.total_epochs - args.start_epoch + 1), 
        eta_min=5e-7
    )

    best_acc = 82.93  # Baseline record to beat

    for epoch in range(args.start_epoch, args.total_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.total_epochs} [DA+FD 200]", leave=False)
        for imgs_sa, imgs_aa, labels in loop:
            imgs_sa, labels = imgs_sa.to(device), labels.to(device)
            imgs_aa = patch_shuffle(imgs_aa).to(device)

            optimizer.zero_grad()

            logits_sa, features_sa = model(imgs_sa)
            logits_aa, _ = model(imgs_aa)

            loss_sa = ce_loss_fn(logits_sa, labels)
            loss_aa = ce_loss_fn(logits_aa, labels)
            loss_fd = model.compute_fd_loss(features_sa)

            loss = loss_sa + loss_aa + args.lambda_fd * loss_fd

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs_sa.size(0)
            preds = logits_sa.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100.0*correct/total:.2f}%", fd=f"{loss_fd.item():.3f}")

        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs_sa, _, val_labels in val_loader:
                imgs_sa, val_labels = imgs_sa.to(device), val_labels.to(device)
                val_outs, _ = model(imgs_sa)
                val_preds = val_outs.argmax(dim=1)
                val_correct += (val_preds == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        train_acc = 100.0 * correct / total
        train_loss_avg = total_loss / total

        logger.info(f"Epoch {epoch}/{args.total_epochs} - Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% (Best: {max(best_acc, val_acc):.2f}%)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'mambavision_da_fd_best.pth')
            logger.info(f"✓ Saved new best DA+FD checkpoint: {best_acc:.2f}%")

    logger.info(f"Extended training finished. Highest Val Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
