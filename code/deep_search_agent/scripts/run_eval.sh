#!/bin/bash
# Evaluation: Compare SFT vs GRPO variants
#
# Usage: bash scripts/run_eval.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo "DeepSearch Agent Evaluation"
echo "============================================"

CONFIG="configs/default.yaml"
MAX_SAMPLES=${MAX_SAMPLES:-500}

python -m evaluation.evaluate \
    --config "$CONFIG" \
    --eval-base \
    --sft-path "./checkpoints/sft" \
    --grpo-path "./checkpoints/grpo/final" \
    --prm-path "./checkpoints/prm" \
    --max-samples "$MAX_SAMPLES"

echo "============================================"
echo "Evaluation complete! Results in ./results/"
echo "============================================"
