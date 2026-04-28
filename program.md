# Autonomous X-ray Research Agent — DGX Spark

## Task
Improve chest X-ray disease classification on ChestMNIST (14 diseases, multi-label).
Metric: **val_auc** (mean AUC-ROC across 14 diseases). Higher is better.
Target: Beat CheXNet benchmark of 0.841.

## Rules
- You may ONLY modify `train.py`
- `prepare.py` is fixed — do not touch it
- Training budget: exactly 5 minutes (TIME_BUDGET=300s in prepare.py)
- Model: DenseNet-121 with ImageNet pretrained weights

## What prepare.py provides
- `make_loaders()` → (train_loader, val_loader, test_loader)
- `evaluate_auc(model, loader, device)` → mean AUC-ROC float
- `TIME_BUDGET=300`, `NUM_CLASSES=14`, `DISEASES=[list of 14]`
- `_default_train_tfm()`, `_default_val_tfm()` — standard ImageNet transforms

## Current best result
See results.tsv for history. Beat the val_auc in the last "keep" row.

## How to write train.py
**IMPORTANT: Use the bash tool to write the file. Do NOT use the edit tool.**

Write the complete new train.py using this exact bash command:
```
cat > /home/nvidia/autoresearch/train.py << 'PYEOF'
[full file content here]
PYEOF
```

## What a good train.py looks like
- Imports from prepare.py
- Builds and trains a model within TIME_BUDGET seconds of actual training time
- Prints exactly these lines at the end (loop.sh parses them):
  ```
  ---
  val_auc:          0.XXXXXX
  training_seconds: XXX.X
  peak_vram_mb:     XXXX.X
  num_steps:        XXXX
  num_params_M:     X.X
  num_epochs:       X
  ```
- Uses `t_train` timer (not wall clock) to measure only training time

## Ideas to try (pick the best one for the current situation)
- Learning rate schedulers: CosineAnnealingLR, OneCycleLR
- Better augmentation: RandomRotation, ColorJitter, CutMix
- Larger batch size (fp16 uses ~2GB, plenty of headroom)
- Freeze early DenseNet layers, train only top layers
- Different backbone: EfficientNet-B3, ResNet-50
- Class-balanced sampling (WeightedRandomSampler)
- Label smoothing in BCEWithLogitsLoss
