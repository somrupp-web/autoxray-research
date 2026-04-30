#!/bin/bash
# ============================================================
# Download NIH ChestX-ray14 dataset to the DGX Spark
#
# Usage:
#   ./download_nih_dataset.sh [destination]
#   Default destination: /home/nvidia/data/NIH_ChestXray14
#
# Space required: ~42 GB images + ~1 GB metadata
# After download, set:  export NIH_CHEST_DIR=<destination>
# Or add to ~/.bashrc:  echo 'export NIH_CHEST_DIR=<destination>' >> ~/.bashrc
#
# Download method: AWS CLI (public bucket, no credentials needed)
#   Install if missing: pip install awscli  or  conda install -c conda-forge awscli
# ============================================================

set -euo pipefail

DEST="${1:-/home/nvidia/data/NIH_ChestXray14}"
S3_BASE="s3://nih-chest-xrays/data"

echo "Destination : $DEST"
echo "Source      : $S3_BASE"
echo "Free space  : $(df -h "$DEST" 2>/dev/null | awk 'NR==2{print $4}' || df -h / | awk 'NR==2{print $4}')"
echo ""

mkdir -p "$DEST"

# Check for aws CLI
if ! command -v aws &>/dev/null; then
    echo "ERROR: aws CLI not found."
    echo "Install: pip install awscli"
    echo ""
    echo "Alternative — Kaggle API:"
    echo "  pip install kaggle"
    echo "  mkdir -p ~/.kaggle && echo '{\"username\":\"YOUR_USER\",\"key\":\"YOUR_KEY\"}' > ~/.kaggle/kaggle.json"
    echo "  kaggle datasets download -d nih-chest-xrays/data -p $DEST --unzip"
    exit 1
fi

# ── Metadata files ─────────────────────────────────────────────────────────
echo "[1/2] Downloading metadata..."
for f in Data_Entry_2017.csv train_val_list.txt test_list.txt BBox_List_2017.csv; do
    echo "  $f"
    aws s3 cp --no-sign-request "$S3_BASE/$f" "$DEST/$f" 2>/dev/null || \
        echo "  WARNING: could not download $f — continuing"
done

# ── Image archives (images_001.tar.gz … images_012.tar.gz) ─────────────────
echo ""
echo "[2/2] Downloading and extracting image archives (~42 GB total)..."
mkdir -p "$DEST/images"

for i in $(seq -f "%03g" 1 12); do
    archive="images_${i}.tar.gz"
    tmp="/tmp/$archive"
    echo ""
    echo "  [$i/012] $archive"

    if aws s3 cp --no-sign-request "$S3_BASE/$archive" "$tmp"; then
        echo "  Extracting..."
        # NIH archives contain images/ subdirectory — strip it so PNGs go to $DEST/images/
        tar -xzf "$tmp" -C "$DEST/images" --strip-components=1 2>/dev/null || \
            tar -xzf "$tmp" -C "$DEST/images" 2>/dev/null || \
            echo "  WARNING: extraction issue with $archive"
        rm -f "$tmp"
    else
        echo "  WARNING: failed to download $archive — skipping"
    fi
done

# ── Verify ─────────────────────────────────────────────────────────────────
N_IMGS=$(find "$DEST/images" -name '*.png' | wc -l)
echo ""
echo "========================================="
echo "Download complete."
echo "Images found : $N_IMGS  (expected ~112,120)"
echo "Dest         : $DEST"
echo ""
echo "To activate NIH dataset for autoresearch:"
echo "  export NIH_CHEST_DIR=$DEST"
echo "  # Or add to ~/.bashrc / tmux session env"
echo ""
echo "In loop.sh, prepend the export before the UV run command, or set it in"
echo "the tmux session launch line inside webui/app.py:"
echo "  NIH_CHEST_DIR=$DEST $UV run train.py ..."
echo "========================================="
