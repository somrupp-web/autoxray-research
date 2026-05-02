# Autonomous X-ray Research Agent — 4-Node DDP

## Task
Improve chest X-ray disease classification on **NIH ChestX-ray14** (112k full-resolution images).
Metric: **val_auc** (mean AUC-ROC across 14 diseases). Higher is better.
Target: Beat CheXNet benchmark of **0.841**.

## Rules
- Use the **Edit tool** to modify ONLY code inside the EXPERIMENT section markers in `train_ddp.py`
- `prepare.py` is fixed — do not touch it
- Everything outside the EXPERIMENT markers is DDP infrastructure — do NOT modify it
- Training is 4-node DDP: each node has 1 GPU, WORLD_SIZE=4, global batch = 4 × per-node batch
- Epochs per iteration are controlled by `MAX_EPOCHS` env var (set by loop_ddp.sh) — do NOT hardcode epochs
- DO NOT rewrite the entire file — edit only the tagged sections

## Editable sections (EXPERIMENT markers)
| Section | Controls |
|---|---|
| `CLASSIFIER_HEAD` | Model architecture — add MLP head, dropout, attention pooling, unfreeze encoder |
| `TRANSFORMS` | Data augmentation — PIL.Image input; **must** end with a tensor (BiomedCLIP preprocessor handles ToTensor) |
| `OPTIMIZER` | Optimizer type, lr, weight_decay, scheduler, criterion / loss function |
| `TRAIN_STEP` | Per-batch forward pass, loss computation, backward, grad clip, optimizer step (AMP goes here) |

## Baseline architecture (in train_ddp.py)
- **BiomedCLIP** ViT-B/16 (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)
- Visual encoder frozen, linear classifier head on top
- AdamW, CosineAnnealingLR, BCEWithLogitsLoss
- Per-node batch 32, global batch 128 (4 nodes × 32)

## What prepare.py provides
- `make_loaders(train_transform, val_transform)` → (train_loader, val_loader, test_loader)
- `evaluate_auc(model, loader, device)` → mean AUC-ROC float (runs on rank-0 only)
- `TIME_BUDGET=600`, `NUM_CLASSES=14`, `DISEASES=[14 disease names]`
- `DATASET_NAME` == `'NIH_ChestXray14_HF'` (HuggingFace dataset, full-resolution NIH)
- `_default_train_tfm()`, `_default_val_tfm()` — standard ImageNet transforms (you can override)

## Dataset facts (NIH ChestX-ray14 HF)
- ~78k train / 11k val / 22k test images (full resolution up to 1024×1024, resized to 224)
- 14 disease classes (multi-label), heavily class-imbalanced
- Already cached on all nodes — no download needed during training
- Each item: `{'image': PIL.Image, 'label': [list of disease strings]}`

## DDP constraints — MUST follow or training crashes

### 1. All ranks must run identical code paths
No rank-conditional logic inside the training loop:
```python
# ✗ WRONG — causes NCCL deadlock
if rank == 0:
    do_something()
# ✓ OK — both branches call a collective
dist.barrier()   # all ranks must hit this
```

### 2. Loss must be scalar before .backward()
```python
loss = criterion(logits, labels).mean()   # ✓
loss = criterion(logits, labels)          # ✗ RuntimeError
```

### 3. DistributedSampler must be set each epoch
```python
train_sampler.set_epoch(epoch)   # ✓ required in loop
```

### 4. Save model as model.module (DDP wraps)
```python
torch.save(model.module.state_dict(), path)   # ✓
torch.save(model.state_dict(), path)          # ✗ saves DDP wrapper
```

### 5. evaluate_auc runs on rank-0 only — no barrier needed inside it
```python
if is_master:
    val_auc = evaluate_auc(model.module, val_loader, device)
```

### 6. dist.barrier() before t0 — already in baseline, keep it
All ranks must sync before the clock starts to avoid timeout skew.

### 7. Print output format — loop_ddp.sh parses with grep
```
val_auc[=: ]+[0-9]+\.[0-9]+
```
The final print block must include:
```python
print('---')
print(f'val_auc:          {val_auc:.6f}')
print(f'training_seconds: {training_seconds:.1f}')
print(f'peak_vram_mb:     {peak_vram:.1f}')
print(f'num_steps:        {step}')
print(f'num_params_M:     {num_params:.1f}')
print(f'num_epochs:       {epoch}')
print(f'world_size:       {world_size}')
print(f'global_batch:     {global_batch}')
```

---

## CRITICAL — Optimizer parameter groups (differential LR)
When splitting parameters into groups, ALWAYS use this exact pattern:
```python
# ✓ CORRECT — n and p are defined by the for clause first
encoder_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
head_params    = [p for n, p in model.named_parameters() if 'classifier' in n]
optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': 1e-5},
    {'params': head_params,    'lr': 1e-4},
], weight_decay=1e-4)

# ✗ WRONG — n is used before the for clause defines it (UnboundLocalError)
encoder_params = [p for p in model.parameters() if 'classifier' not in n for n, p in model.named_parameters()]
```

### CRITICAL — Overlapping parameter groups crash with ValueError
BiomedCLIP's visual encoder has an internal `.head` attribute. If your classifier is named
`self.head`, then parameter names like `module.encoder.head.weight` contain BOTH 'encoder'
AND 'head' — causing them to appear in two groups.

```python
# ✗ WRONG — BiomedCLIP encoder has internal .head, so 'encoder.head.*' matches both filters
encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n]
head_params    = [p for n, p in model.named_parameters() if 'head' in n]
# → ValueError: some parameters appear in more than one parameter group

# ✓ CORRECT — use startswith or mutually exclusive filter
head_params    = [p for n, p in model.named_parameters() if n.startswith('module.head')]
encoder_params = [p for n, p in model.named_parameters() if not n.startswith('module.head')]
```

**Rule:** always make param group filters mutually exclusive. One group catches what the
other misses — never let both conditions be true for the same parameter name.

---

## CRITICAL — These mistakes crash DDP training

### AMP (mixed precision) with DDP
```python
# ✓ Correct AMP + DDP
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = criterion(model(imgs), labels).mean()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

### Gradient accumulation with DDP
```python
# ✓ Use model.no_sync() for all but last accumulation step
for i, (imgs, labels) in enumerate(train_loader):
    ctx = model.no_sync() if (i + 1) % accum_steps != 0 else contextlib.nullcontext()
    with ctx:
        loss = criterion(model(imgs), labels).mean() / accum_steps
        loss.backward()
    if (i + 1) % accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Ideas to try (pick one not already in results.tsv)

### Augmentation (NIH data is full resolution — heavier aug is safe)
- RandAugment, CutMix, MixUp, stronger ColorJitter + RandomRotation
- Random erasing, GridDistortion
- Center crop + random crop (vs. simple resize)

### Fine-tuning strategy
- Unfreeze last N transformer blocks of BiomedCLIP encoder (with lower LR)
- Differential LR: encoder 1e-5, classifier 1e-4
- Layer-wise LR decay (LLRD)

### Classifier head
- Multi-layer MLP head instead of single linear: Linear → ReLU → Dropout → Linear
- Attention pooling instead of CLS token

### Optimizer / scheduler
- AdamW with different β1/β2, Lion optimizer
- OneCycleLR, WarmupCosine, ReduceLROnPlateau
- Larger effective batch via gradient accumulation + proportional LR

### Loss / class imbalance
- Focal loss (γ=2) for rare diseases
- Asymmetric loss (ASL) for multi-label
- pos_weight in BCEWithLogitsLoss (inverse class frequency)
- Label smoothing (ε=0.1)

### Mixed precision
- torch.cuda.amp (AMP) — significant speedup on GB10 GPU

### Regularization
- Dropout in classifier (p=0.3-0.5)
- Stochastic Depth / DropPath in encoder
- Weight decay tuning
