# Experiment 01: Standard MHA + Absolute Position Embeddings (baseline)
# This is the original nanoGPT architecture.
#
# Run: ~/venv/bin/python train_fsdp.py config/experiments/exp01_mha_abspe.py

exec(open('config/experiments/base.py').read())

out_dir       = 'out_experiments/exp01_mha_abspe'
n_kv_head     = 0        # MHA: n_kv_head == n_head
wandb_run_name = 'exp01-mha-abspe'
