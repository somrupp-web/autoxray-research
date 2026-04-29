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

## Current best result
See results.tsv for history. Beat the val_auc in the highest-scoring "keep" row.

## How to write train.py
**IMPORTANT: Use the bash tool to write the file. Do NOT use the edit tool.**

Write the complete new train.py using this exact bash command:
```
cat > /home/nvidia/autoresearch/train.py << 'PYEOF'
[full file content here]
PYEOF
```

---

## CRITICAL — These mistakes crash training (loop skips the iteration)

### 1. Loss must be a scalar before .backward()
PyTorch's `.backward()` requires a single number. BCEWithLogitsLoss with
`reduction='none'` returns shape `(batch, 14)` — NOT a scalar.

```python
# WRONG — crashes with: RuntimeError: grad can be implicitly created only for scalar outputs
loss = criterion(logits, labels)
scaler.scale(loss).backward()

# CORRECT — always reduce to scalar first
loss = criterion(logits, labels).mean()
scaler.scale(loss).backward()
```

This applies to ALL loss functions: BCEWithLogitsLoss, FocalLoss, custom losses.
If you implement a custom Focal Loss or weighted loss, its `forward()` must return
a scalar OR you must call `.mean()` / `.sum()` on its output.

### 2. Print format must match exactly
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

If val_auc is printed differently (e.g. `validation_auc`, `auc=`, inside a dict),
the loop will not find it and will skip the iteration.

### 3. WeightedRandomSampler — labels are already tensors
When computing class weights, medmnist labels are already numpy arrays or tensors.
Do NOT call `torch.tensor()` on them — it crashes.

```python
# WRONG — crashes: ValueError: only one element tensors can be converted to Python scalars
labels_tensor = torch.tensor(labels)

# CORRECT — use numpy stack
import numpy as np
labels_array = np.stack([lbl.numpy().squeeze() for _, lbl in train_loader.dataset])
# labels_array shape: (N, 14) — now compute class weights from this
pos_freq = labels_array.mean(axis=0).clip(1e-6, 1 - 1e-6)  # (14,)
weights_per_sample = (1.0 / pos_freq[None, :] * labels_array +
                      1.0 / (1 - pos_freq)[None, :] * (1 - labels_array)).mean(axis=1)
sampler = torch.utils.data.WeightedRandomSampler(weights_per_sample, len(weights_per_sample))
```

### 4. Model must be saved for test inference
Always keep this line at the end of training:
```python
_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model.pth')
torch.save(model.state_dict(), _model_path)
```

---

## What a good train.py looks like
- Imports from prepare.py (TIME_BUDGET, NUM_CLASSES, make_loaders, evaluate_auc)
- Trains within TIME_BUDGET seconds using `t_train` timer (not wall clock)
- Uses mixed precision (GradScaler + autocast) for speed
- Evaluates with `evaluate_auc(model, val_loader, device)` — do not reimplement this
- Saves model weights to `trained_model.pth`
- Prints the exact output block above

## Ideas to try (pick one not already in results.tsv)
- **Schedulers**: CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, WarmupCosine
- **Optimizers**: AdamW with different lr/weight_decay, SGD+momentum, Lion
- **Augmentation**: RandAugment, CutMix, MixUp, stronger ColorJitter + RandomRotation
- **Architecture**: EfficientNet-B3, ResNet-50, DenseNet-169 (larger than DenseNet-121)
- **Training tricks**: Gradient clipping, SWA (Stochastic Weight Averaging), EMA
- **Regularisation**: Label smoothing, Dropout tuning, DropBlock
- **Sampling**: WeightedRandomSampler for class imbalance
- **Focal loss**: Replace BCE — but remember `.mean()` on the output!
- **Batch size**: Larger batch (128, 256) with proportionally scaled LR
