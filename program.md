# Autonomous X-ray Research Agent — DGX Spark Cluster

## Task
Improve chest X-ray disease classification on ChestMNIST (14 diseases, multi-label).
Metric: **val_auc** (mean AUC-ROC across 14 diseases). Higher is better.
Target: Beat CheXNet benchmark of 0.841.

## Cluster context
You are one of **4 agents running in parallel** on a 4-node DGX Spark cluster.
The `results.tsv` file contains experiments from **all nodes**.
Before choosing your improvement, scan results.tsv for techniques already tried.
Pick something different — parallel agents should explore diverse directions.

## Rules
- You may ONLY modify `train.py`
- `prepare.py` is fixed — do not touch it
- Training budget: exactly 10 minutes (TIME_BUDGET=600s in prepare.py)
- Model resets to ImageNet pretrained weights at the start of every run

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

## Ideas to try — assign by node to avoid duplication
- **Schedulers**: CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau, WarmupCosine
- **Optimizers**: AdamW, SGD+momentum, Lion
- **Augmentation**: RandAugment, CutMix, MixUp, TrivialAugment
- **Architecture**: EfficientNet-B3/B4, ResNet-50, ViT-Small, DenseNet-169
- **Training tricks**: Gradient clipping, SWA (Stochastic Weight Averaging), EMA
- **Regularisation**: Mixup + label smoothing, Dropout tuning, DropBlock
- **Sampling**: WeightedRandomSampler, focal loss instead of BCE
- **Batch size**: larger batch (128, 256) with scaled LR
