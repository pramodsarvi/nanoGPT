#!/bin/bash
# Run exp11 (GQA) and exp12 (MHA) sequentially.
# Make sure data is prepared first:
#   python data/fineweb_edu/prepare.py
#
# Usage: bash config/experiments/run_exp11_exp12.sh

set -e  # exit on any error

PYTHON=${PYTHON:-python}
TRAIN=train_fsdp.py

echo "============================================"
echo "Starting exp11: GQA n_kv_head=4 from scratch"
echo "============================================"
$PYTHON $TRAIN config/experiments/exp11_gqa4_fineweb_scratch.py

# echo "============================================"
# echo "Starting exp12: MHA baseline from scratch"
#     echo "============================================"
#     $PYTHON $TRAIN config/experiments/exp12_mha_fineweb_scratch.py

echo "============================================"
echo "Both experiments done. Compare results:"
echo "  $PYTHON config/experiments/compare.py --plot"
echo "============================================"
