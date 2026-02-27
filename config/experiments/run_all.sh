#!/bin/bash
# Run all experiments sequentially and collect results.
# Each writes its own out_experiments/<name>/log.csv
#
# Usage: bash config/experiments/run_all.sh
# Or run individual ones:
#   ~/venv/bin/python train_fsdp.py config/experiments/exp01_mha_abspe.py

set -e
PYTHON=~/venv/bin/python

experiments=(
    config/experiments/exp01_mha_abspe.py
    config/experiments/exp02_gqa2_rope.py
    config/experiments/exp03_gqa3_rope.py
    config/experiments/exp04_mqa_rope.py
    config/experiments/exp05_gqa2_rope500k.py
)

for cfg in "${experiments[@]}"; do
    echo "========================================"
    echo "Running: $cfg"
    echo "========================================"
    $PYTHON train_fsdp.py "$cfg"
done

echo ""
echo "All experiments done. Compare results:"
echo "  $PYTHON config/experiments/compare.py"
