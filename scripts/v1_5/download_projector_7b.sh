#!/bin/bash

# Download pretrained mm_projector for LLaVA-v1.5-7B (Stage 1 output)
# This projector is required before running Stage 2 instruction finetuning.

set -e

PROJECTOR_DIR="./checkpoints/llava-v1.5-7b-pretrain"
PROJECTOR_FILE="$PROJECTOR_DIR/mm_projector.bin"
REPO_ID="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5"

if [ -f "$PROJECTOR_FILE" ]; then
    echo "[SKIP] mm_projector.bin already exists at $PROJECTOR_FILE"
    ls -lh "$PROJECTOR_FILE"
    exit 0
fi

echo "Downloading pretrained projector for LLaVA-v1.5-7B..."
echo "  From: $REPO_ID"
echo "  To:   $PROJECTOR_DIR"

mkdir -p "$PROJECTOR_DIR"

python -c "
from huggingface_hub import hf_hub_download
import os

filepath = hf_hub_download(
    repo_id='$REPO_ID',
    filename='mm_projector.bin',
    local_dir='$PROJECTOR_DIR',
    local_dir_use_symlinks=False
)
print(f'Downloaded to: {filepath}')
"

if [ -f "$PROJECTOR_FILE" ]; then
    echo ""
    echo "============================================"
    echo "  Download completed successfully!"
    echo "  $(ls -lh "$PROJECTOR_FILE")"
    echo "============================================"
else
    echo "[ERROR] Download failed, file not found: $PROJECTOR_FILE"
    exit 1
fi
