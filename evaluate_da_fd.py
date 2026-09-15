import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score

from models.mambavision_da_fd import MambaVisionDAFD

class SoyAgeingDataset(Dataset):
    def __init__(self, anno_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
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
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = '/home/subsea/reena/intern/VMambaFDCL/data/SoyAgeing-R1/R1'
    img_dir = os.path.join(base_dir, 'images')
    test_anno = os.path.join(base_dir, 'anno', 'test.txt')

    eval_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_set = SoyAgeingDataset(test_anno, img_dir, transform=eval_transform)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4)

    model = MambaVisionDAFD(num_classes=198).to(device)
    model.load_state_dict(torch.load('mambavision_da_fd_best.pth', map_location=device))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    print(f"Evaluating {len(test_set)} test samples across 198 classes on DA+FD checkpoint...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs, _ = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    top1 = accuracy_score(all_labels, all_preds) * 100.0
    top5 = top_k_accuracy_score(all_labels, all_probs, k=5, labels=np.arange(198)) * 100.0

    print("\n" + "="*45)
    print(f" Top-1 Accuracy : {top1:.2f}%")
    print(f" Top-5 Accuracy : {top5:.2f}%")
    print("="*45 + "\n")

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    print(f"Macro Precision    : {report['macro avg']['precision'] * 100.0:.2f}%")
    print(f"Macro Recall       : {report['macro avg']['recall'] * 100.0:.2f}%")
    print(f"Macro F1-Score     : {report['macro avg']['f1-score'] * 100.0:.2f}%")
    print(f"Weighted F1-Score  : {report['weighted avg']['f1-score'] * 100.0:.2f}%")

if __name__ == '__main__':
    main()
