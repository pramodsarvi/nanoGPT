#!/bin/bash
# Run exp11 (GQA) and exp12 (MHA) sequentially using FSDP.
# Make sure data is prepared first:
#   python data/fineweb_edu/prepare.py
#
# Usage:
#   bash config/experiments/run_exp11_exp12.sh          # 4 GPUs (default)
#   NGPUS=2 bash config/experiments/run_exp11_exp12.sh  # 2 GPUs
#   NGPUS=1 bash config/experiments/run_exp11_exp12.sh  # 1 GPU (single process, no FSDP)

set -e  # exit on any error

NGPUS=${NGPUS:-4}
TRAIN=train_fsdp.py

if [ "$NGPUS" -gt 1 ]; then
    LAUNCHER="torchrun --standalone --nproc_per_node=$NGPUS"
else
    LAUNCHER="${PYTHON:-python}"
fi

echo "============================================"
echo "Starting exp11: GQA n_kv_head=4 from scratch ($NGPUS GPUs)"
echo "============================================"
$LAUNCHER $TRAIN config/experiments/exp11_gqa4_fineweb_scratch.py

# echo "============================================"
# echo "Starting exp12: MHA baseline from scratch"
#     echo "============================================"
#     $PYTHON $TRAIN config/experiments/exp12_mha_fineweb_scratch.py

echo "============================================"
echo "Both experiments done. Compare results:"
echo "  $PYTHON config/experiments/compare.py --plot"
echo "============================================"
