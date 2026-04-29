# Autonomous X-ray Research Agent — DGX Spark

## Task
Improve chest X-ray disease classification on ChestMNIST (14 diseases, multi-label).
Metric: **val_auc** (mean AUC-ROC across 14 diseases). Higher is better.
Target: Beat CheXNet benchmark of 0.841.

## Rules
- You may ONLY modify `train.py`
- `prepare.py` is fixed — do not touch it
- Training budget: exactly 10 minutes (TIME_BUDGET=600s in prepare.py)
- Model resets to ImageNet pretrained DenseNet-121 at the start of every run

## What prepare.py provides
- `make_loaders()` → (train_loader, val_loader, test_loader)
- `evaluate_auc(model, loader, device)` → mean AUC-ROC float
- `TIME_BUDGET=600`, `NUM_CLASSES=14`, `DISEASES=[list of 14]`
- `_default_train_tfm()`, `_default_val_tfm()` — standard ImageNet transforms
- `train_loader.dataset` — the underlying PyTorch Dataset object
- No internal classes like `_XrayDataset` are exposed — do not try to import them

## Dataset facts
- ChestMNIST: 78k train / 11k val / 22k test images, 28×28 upsampled to 224×224
- Each label is a numpy array of shape `(14,)` or `(14,1)` — call `.squeeze()` for `(14,)`
- To access raw labels for weight computation, use `medmnist.ChestMNIST` directly:
  ```python
  from medmnist import ChestMNIST
  ds = ChestMNIST(split='train', size=28, download=True)
  # ds[i] returns (img, lbl) where lbl is numpy array shape (14,) or (14,1)
  ```
- To rebuild train_loader with a custom sampler:
  ```python
  train_loader, val_loader, test_loader = make_loaders()
  train_loader = torch.utils.data.DataLoader(
      train_loader.dataset, batch_size=32, sampler=your_sampler,
      num_workers=train_loader.num_workers, pin_memory=True, drop_last=True)
  ```

---

## CRITICAL — These mistakes crash training (loop skips the iteration)

### 1. Loss must be a scalar before .backward()
`.backward()` requires a single number. BCEWithLogitsLoss with `reduction='none'`
returns shape `(batch, 14)` — NOT a scalar. Always reduce first:

```python
loss = criterion(logits, labels).mean()   # ✓ scalar
loss = criterion(logits, labels)          # ✗ crashes: RuntimeError: grad can be
                                          #   implicitly created only for scalar outputs
```

Applies to ALL loss functions including custom ones.

### 2. Do not call torch.tensor() on existing tensors or numpy arrays
Labels from the DataLoader are already tensors. Labels from medmnist are numpy arrays.
Both crash with `ValueError: only one element tensors can be converted to Python scalars`.

```python
torch.tensor(label)          # ✗ crashes if label is already a tensor/array
np.array(label)              # ✓ safe conversion from tensor
label.numpy().squeeze()      # ✓ tensor → numpy
```

### 3. Print format must match exactly
loop.sh parses val_auc with: `grep -Eo 'val_auc[=: ]+[0-9]+\.[0-9]+'`

The final print block must be:
```python
print('---')
print(f'val_auc:          {val_auc:.6f}')
print(f'training_seconds: {training_seconds:.1f}')
print(f'peak_vram_mb:     {peak_vram:.1f}')
print(f'num_steps:        {step}')
print(f'num_params_M:     {num_params:.1f}')
print(f'num_epochs:       {epoch}')
```

### 4. Model must be saved for test inference
```python
_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model.pth')
torch.save(model.state_dict(), _model_path)
```

---

## Ideas to try (pick one not already in results.tsv)
- **Schedulers**: CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, WarmupCosine
- **Optimizers**: AdamW with different lr/weight_decay, SGD+momentum, Lion
- **Augmentation**: RandAugment, CutMix, MixUp, stronger ColorJitter + RandomRotation
- **Architecture**: EfficientNet-B3, ResNet-50, DenseNet-169
- **Training tricks**: Gradient clipping, SWA, EMA
- **Regularisation**: Label smoothing, Dropout tuning, DropBlock
- **Class imbalance**: WeightedRandomSampler, pos_weight in loss, focal loss
- **Batch size**: Larger batch (64, 128) with proportionally scaled LR
