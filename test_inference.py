"""
test_inference.py — Run inference on a fixed ChestMNIST test sample.

Loads trained_model.pth, runs inference on test index 42, saves:
  - test_xray.png              : upscaled 224x224 grayscale PNG
  - test_inference_results.json: structured prediction results
"""

import json
import os
import sys
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torchvision.models import densenet121
from medmnist import ChestMNIST
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(REPO, "trained_model.pth")
XRAY_OUT   = os.path.join(REPO, "test_xray.png")
JSON_OUT   = os.path.join(REPO, "test_inference_results.json")

# ── Constants from prepare.py ─────────────────────────────────────────────────
from prepare import DISEASES, NUM_CLASSES, _default_val_tfm

TEST_IDX   = 42
THRESHOLD  = 0.5

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Build DenseNet-121 with 14-class head and load saved weights."""
    if not os.path.exists(MODEL_PATH):
        error = {"error": "no model"}
        print(json.dumps(error))
        sys.exit(1)

    model = densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_sample(idx: int):
    """Return (raw PIL image 28x28 grayscale, transformed tensor, label array)."""
    dataset = ChestMNIST(split="test", size=28, download=True)
    img_pil, label = dataset[idx]          # img_pil is PIL, label is ndarray shape (14,)

    # Save raw grayscale image (img_pil may be L or RGB from medmnist)
    # Ensure grayscale
    img_gray = img_pil.convert("L")

    # Build transformed tensor for model input (val transform expects RGB)
    transform = _default_val_tfm()
    img_rgb = img_pil.convert("RGB")
    img_tensor = transform(img_rgb).unsqueeze(0)   # (1, 3, 224, 224)

    return img_gray, img_tensor, label.squeeze()   # label shape (14,)


def save_xray(img_gray: Image.Image):
    """Upscale 28x28 to 224x224 with LANCZOS and save as grayscale PNG."""
    img_upscaled = img_gray.resize((224, 224), Image.LANCZOS)
    img_upscaled.save(XRAY_OUT)


def run_inference(model, img_tensor):
    """Return sigmoid probabilities as a Python list of floats."""
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        logits = model(img_tensor)          # (1, 14)
        probs  = torch.sigmoid(logits)      # (1, 14)
    return probs.squeeze().cpu().tolist()   # list of 14 floats


def build_results(probs, labels):
    """Assemble and return the results dict."""
    predictions = []
    correct_count = 0

    for i, disease in enumerate(DISEASES):
        confidence = round(float(probs[i]), 4)
        actual     = int(labels[i])
        predicted  = 1 if confidence >= THRESHOLD else 0
        correct    = predicted == actual
        if correct:
            correct_count += 1

        predictions.append({
            "disease":    disease,
            "confidence": confidence,
            "predicted":  predicted,
            "actual":     actual,
            "correct":    correct,
        })

    wrong = NUM_CLASSES - correct_count

    return {
        "test_idx":              TEST_IDX,
        "timestamp":             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "val_auc_from_training": None,
        "predictions":           predictions,
        "summary": {
            "total":   NUM_CLASSES,
            "correct": correct_count,
            "wrong":   wrong,
        },
    }


def main():
    model              = load_model()
    img_gray, img_tensor, labels = load_sample(TEST_IDX)

    save_xray(img_gray)

    probs   = run_inference(model, img_tensor)
    results = build_results(probs, labels)

    # Write JSON file
    with open(JSON_OUT, "w") as f:
        json.dump(results, f, indent=2)

    # Print to stdout
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
