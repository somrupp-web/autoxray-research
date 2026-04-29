"""
test_inference.py — Run inference on a ChestMNIST test sample.

Usage:
    uv run test_inference.py [--val-auc 0.775] [--iter 3] [--idx 42]

The sample index is resolved in this order:
  1. --idx CLI argument
  2. test_xray_idx.txt in the repo directory (written by WebUI "New Sample" button)
  3. Default: 42

Saves / appends:
  - test_xray.png                 : 224x224 grayscale PNG
  - test_inference_results.json   : latest result
  - test_inference_history.json   : append-only list of all results across runs
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torchvision.models import densenet121
from medmnist import ChestMNIST
from PIL import Image

from prepare import DISEASES, NUM_CLASSES, _default_val_tfm

# ── Paths ──────────────────────────────────────────────────────────────────
REPO         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(REPO, "trained_model.pth")
XRAY_OUT     = os.path.join(REPO, "test_xray.png")
JSON_OUT     = os.path.join(REPO, "test_inference_results.json")
HISTORY_PATH = os.path.join(REPO, "test_inference_history.json")
IDX_FILE     = os.path.join(REPO, "test_xray_idx.txt")

DEFAULT_IDX = 42
THRESHOLD   = 0.5
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val-auc", type=float, default=None,
                   help="val_auc from the training run that produced this model")
    p.add_argument("--iter",    type=int,   default=0,
                   help="loop iteration number")
    p.add_argument("--idx",     type=int,   default=None,
                   help="ChestMNIST test image index (overrides test_xray_idx.txt)")
    return p.parse_args()


def resolve_idx(args_idx):
    """CLI arg > test_xray_idx.txt > DEFAULT_IDX (42)."""
    if args_idx is not None:
        return args_idx
    try:
        with open(IDX_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return DEFAULT_IDX


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(json.dumps({"error": "no model"}))
        sys.exit(1)
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model


def load_sample(idx: int):
    # High-res (224x224) for display
    img_hires, label = ChestMNIST(split="test", size=224, download=True)[idx]
    img_gray = img_hires.convert("L")

    # 28x28 for inference — matches training distribution
    img_lores, _ = ChestMNIST(split="test", size=28, download=True)[idx]
    img_tensor = _default_val_tfm()(img_lores.convert("RGB")).unsqueeze(0)

    return img_gray, img_tensor, label.squeeze()


def save_xray(img_gray):
    img_gray.save(XRAY_OUT)


def run_inference(model, img_tensor):
    with torch.no_grad():
        probs = torch.sigmoid(model(img_tensor.to(device)))
    return probs.squeeze().cpu().tolist()


def build_results(probs, labels, val_auc, iteration, test_idx):
    predictions = []
    correct = 0
    for i, disease in enumerate(DISEASES):
        conf      = round(float(probs[i]), 4)
        actual    = int(labels[i])
        predicted = 1 if conf >= THRESHOLD else 0
        ok        = predicted == actual
        if ok:
            correct += 1
        predictions.append({
            "disease":    disease,
            "confidence": conf,
            "predicted":  predicted,
            "actual":     actual,
            "correct":    ok,
        })
    return {
        "iteration":             iteration,
        "val_auc_from_training": val_auc,
        "test_correct":          correct,
        "test_wrong":            NUM_CLASSES - correct,
        "test_accuracy_pct":     round(correct / NUM_CLASSES * 100, 1),
        "test_idx":              test_idx,
        "timestamp":             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predictions":           predictions,
        "summary": {"total": NUM_CLASSES, "correct": correct, "wrong": NUM_CLASSES - correct},
    }


def append_history(result):
    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = []
    except Exception:
        history = []
    history.append(result)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def main():
    args     = parse_args()
    test_idx = resolve_idx(args.idx)

    model  = load_model()
    img_gray, img_tensor, labels = load_sample(test_idx)

    save_xray(img_gray)

    probs   = run_inference(model, img_tensor)
    results = build_results(probs, labels, args.val_auc, args.iter, test_idx)

    with open(JSON_OUT, "w") as f:
        json.dump(results, f, indent=2)

    append_history(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
