"""
train.py — X-ray disease classification.
Experiment: OneCycleLR scheduler + stronger data augmentation.
Change: OneCycleLR with warmup provides better convergence; RandomRotation and ColorJitter improve generalization.
"""

import os, gc, time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR

from prepare import (
    TIME_BUDGET, NUM_CLASSES, DISEASES,
    make_loaders, evaluate_auc,
    _default_train_tfm, _default_val_tfm,
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LR           = 5e-4
MAX_LR       = 1e-2
WEIGHT_DECAY = 1e-5
LABEL_SMOOTH = 0.1

# ── Stronger augmentation ─────────────────────────────────────────────────────
def _strong_train_tfm():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

# ── Setup ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader, test_loader = make_loaders()

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.densenet121(weights='IMAGENET1K_V1')
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model = model.to(device)
num_params = sum(p.numel() for p in model.parameters()) / 1e6

# ── Optimizer + scheduler + mixed precision scaler ───────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# OneCycleLR: warmup + high LR + annealing
scheduler = OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    steps_per_epoch=len(train_loader),
    epochs=int(TIME_BUDGET / 300) + 1,
    pct_start=0.2,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0
)
scaler    = GradScaler()
criterion = nn.BCEWithLogitsLoss(reduction='none')

# ── Training loop — fixed TIME_BUDGET with fp16 ───────────────────────────────
t_start   = time.time()
t_train   = 0.0
step      = 0
epoch     = 0
peak_vram = 0.0

print('Training started (fp16 + OneCycleLR + strong augmentation)...')
while True:
    epoch += 1
    model.train()
    for imgs, labels in train_loader:
        t0 = time.time()
        if t_train >= TIME_BUDGET:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            soft_labels = labels * (1 - LABEL_SMOOTH) + LABEL_SMOOTH / NUM_CLASSES
            loss = criterion(logits, soft_labels).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        t_train += time.time() - t0
        step += 1
        if device.type == 'cuda':
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1024**2)
    if t_train >= TIME_BUDGET:
        break

training_seconds = t_train
total_seconds    = time.time() - t_start

val_auc = evaluate_auc(model, val_loader, device)

# Save model weights for test inference
_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model.pth')
torch.save(model.state_dict(), _model_path)

print('---')
print(f'val_auc:          {val_auc:.6f}')
print(f'training_seconds: {training_seconds:.1f}')
print(f'total_seconds:    {total_seconds:.1f}')
print(f'peak_vram_mb:     {peak_vram:.1f}')
print(f'num_steps:        {step}')
print(f'num_params_M:     {num_params:.1f}')
print(f'num_epochs:       {epoch}')
