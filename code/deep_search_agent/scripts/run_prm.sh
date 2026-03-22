#!/bin/bash
# Phase 2: PRM Training
# Collects rollouts from SFT model and trains Intent-Aware PRM
#
# Resource: Single GPU, ~30 minutes
# Usage: bash scripts/run_prm.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 2: Intent-Aware PRM Training"
echo "============================================"

CONFIG="configs/default.yaml"
DATA_PATH="./data/prm_rollouts.json"
SFT_MODEL="./checkpoints/sft"

# Step 1: Collect rollout trajectories from SFT model
echo "[1/2] Collecting rollout trajectories..."
python -m training.prm_trainer \
    --config "$CONFIG" \
    --collect-only \
    --data-path "$DATA_PATH" \
    --sft-model-path "$SFT_MODEL"

# Step 2: Train PRM
echo "[2/2] Training Intent-Aware PRM..."
python -m training.prm_trainer \
    --config "$CONFIG" \
    --train-only \
    --data-path "$DATA_PATH"

echo "============================================"
echo "PRM training complete!"
echo "Model saved to ./checkpoints/prm/"
echo "============================================"
