#!/bin/bash
# Phase 1: SFT Warmstart
# Generates demo trajectories and fine-tunes Qwen2.5-3B with LoRA
#
# Resource: Single GPU, ~1 hour
# Usage: bash scripts/run_sft.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 1: SFT Warmstart"
echo "============================================"

# Config
CONFIG="configs/default.yaml"
DATA_PATH="./data/sft_trajectories.json"
MAX_SAMPLES=${MAX_SAMPLES:-2000}

# Step 1: Generate demonstration trajectories
echo "[1/2] Generating demonstration trajectories..."
python -m training.sft_warmstart \
    --config "$CONFIG" \
    --generate-only \
    --data-path "$DATA_PATH" \
    --max-samples "$MAX_SAMPLES"

# Step 2: SFT training
echo "[2/2] Running SFT training..."
python -m training.sft_warmstart \
    --config "$CONFIG" \
    --train-only \
    --data-path "$DATA_PATH"

echo "============================================"
echo "SFT training complete!"
echo "Model saved to ./checkpoints/sft/"
echo "============================================"
