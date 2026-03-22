#!/bin/bash
# Phase 3: GRPO RL Training
# Trains the agent with Group Relative Policy Optimization
# Uses IA-PRM for process rewards + F1 for outcome rewards
#
# Resource: 4×A100 40G, ~3-5 hours
# Usage: bash scripts/run_grpo.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo "Phase 3: GRPO Training"
echo "============================================"

CONFIG="configs/default.yaml"
SFT_MODEL="./checkpoints/sft"
PRM_MODEL="./checkpoints/prm"
NUM_EPISODES=${NUM_EPISODES:-500}

# Multi-GPU with accelerate
accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m training.grpo_trainer \
    --config "$CONFIG" \
    --sft-model-path "$SFT_MODEL" \
    --prm-path "$PRM_MODEL" \
    --num-episodes "$NUM_EPISODES"

echo "============================================"
echo "GRPO training complete!"
echo "Model saved to ./checkpoints/grpo/"
echo "============================================"
