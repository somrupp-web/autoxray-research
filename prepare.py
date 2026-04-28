"""
prepare.py — Fixed X-ray data prep and evaluation utilities.
DO NOT MODIFY. The agent only modifies train.py.

Provides:
  - ChestMNIST data loaders (78k train / 11k val / 22k test)
  - evaluate_auc()  — ground truth metric, called from train.py
  - TIME_BUDGET     — fixed 5-minute wall-clock training budget
  - DISEASES        — list of 14 disease class names
"""

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from medmnist import ChestMNIST
import numpy as np
from sklearn.metrics import roc_auc_score

# ── Fixed constants (do not modify) ──────────────────────────────────────────
TIME_BUDGET  = 600          # training seconds — 10 minutes
BATCH_SIZE   = 32
NUM_WORKERS  = 4
IMAGE_SIZE   = 224
NUM_CLASSES  = 14

DISEASES = [
    'atelectasis', 'cardiomegaly', 'effusion', 'infiltration',
    'mass', 'nodule', 'pneumonia', 'pneumothorax',
    'consolidation', 'edema', 'emphysema', 'fibrosis',
    'pleural', 'hernia'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Dataset wrapper ───────────────────────────────────────────────────────────
class _XrayDataset(torch.utils.data.Dataset):
    def __init__(self, split, transform):
        self.base = ChestMNIST(split=split, size=28, download=True)
        self.transform = transform

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label.squeeze(), dtype=torch.float32)

# ── Public API ────────────────────────────────────────────────────────────────
def make_loaders(train_transform=None, val_transform=None):
    """Return (train_loader, val_loader, test_loader). Agent provides transforms."""
    if train_transform is None:
        train_transform = _default_train_tfm()
    if val_transform is None:
        val_transform = _default_val_tfm()

    train_loader = DataLoader(
        _XrayDataset('train', train_transform),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(
        _XrayDataset('val', val_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(
        _XrayDataset('test', val_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader

def evaluate_auc(model, loader, device):
    """Ground-truth evaluation — do not modify. Returns mean AUC-ROC across 14 diseases."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.numpy())
    preds  = np.concatenate(all_preds,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    aucs = []
    for i in range(NUM_CLASSES):
        if labels[:, i].sum() > 0:
            aucs.append(roc_auc_score(labels[:, i], preds[:, i]))
    return float(np.mean(aucs))

def _default_train_tfm():
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _default_val_tfm():
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
