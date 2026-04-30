#!/bin/bash
# Run on each cluster node to download NIH ChestX-ray14 from AWS Open Data
set -euo pipefail

DEST=/home/nvidia/data/NIH_ChestXray14
AWS=/home/nvidia/autoresearch/.venv/bin/aws
S3=s3://nih-chest-xrays/data
LOG=$DEST/download.log

mkdir -p "$DEST/images"
echo "[$(date +%T)] Starting NIH ChestX-ray14 download" | tee "$LOG"

# Metadata files
for f in Data_Entry_2017.csv train_val_list.txt test_list.txt; do
    echo "[$(date +%T)] Fetching $f" | tee -a "$LOG"
    $AWS s3 cp --no-sign-request "$S3/$f" "$DEST/$f" >> "$LOG" 2>&1 || \
        echo "WARNING: $f failed" | tee -a "$LOG"
done

# 12 image archives (~3.5 GB each)
for i in 001 002 003 004 005 006 007 008 009 010 011 012; do
    archive="images_${i}.tar.gz"
    tmp="/tmp/$archive"
    echo "[$(date +%T)] Downloading $archive" | tee -a "$LOG"
    $AWS s3 cp --no-sign-request "$S3/$archive" "$tmp" >> "$LOG" 2>&1
    echo "[$(date +%T)] Extracting $archive" | tee -a "$LOG"
    tar -xzf "$tmp" -C "$DEST/images" --strip-components=1 2>>"$LOG" || \
        tar -xzf "$tmp" -C "$DEST/images" 2>>"$LOG" || \
        echo "WARNING: extraction issue $archive" | tee -a "$LOG"
    rm -f "$tmp"
    n=$(find "$DEST/images" -name "*.png" | wc -l)
    echo "[$(date +%T)] Done $archive | total images so far: $n" | tee -a "$LOG"
done

N=$(find "$DEST/images" -name "*.png" | wc -l)
echo "[$(date +%T)] DOWNLOAD COMPLETE: $N images (expected ~112120)" | tee -a "$LOG"
