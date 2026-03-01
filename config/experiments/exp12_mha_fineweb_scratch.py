# Experiment 12: GPT-2 Small from scratch with MHA (baseline) on FineWeb-Edu
# Direct comparison baseline for exp11 (GQA n_kv_head=4).
# Identical architecture and hyperparams — only difference is MHA vs GQA.
#
# Data prep (run once on A100, ~5 min for 1BT):
#   python data/fineweb_edu/prepare.py --max_tokens 1_000_000_000
#
# Run:
#   python train_fsdp.py config/experiments/exp12_mha_fineweb_scratch.py

import os as _os
exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp12_mha_fineweb_scratch'
dataset        = 'fineweb_edu'
init_from      = 'resume' if _os.path.exists('out_experiments/exp12_mha_fineweb_scratch/ckpt.pt') else 'scratch'
n_kv_head      = 0              # MHA baseline — standard full attention
use_rope       = True
wandb_run_name = 'exp12-mha-fineweb-scratch'

# GPT-2 Small architecture — identical to exp11
n_layer    = 12
n_head     = 12
n_embd     = 768
block_size = 1024
batch_size = 32
gradient_accumulation_steps = 4   # effective batch = 128
dropout    = 0.0
compile    = True

# Identical schedule to exp11
max_iters      = 50000
eval_interval  = 500
warmup_iters   = 500
learning_rate  = 6e-4
min_lr         = 6e-5
lr_scheduler   = 'cosine'
lr_decay_iters = 50000
weight_decay   = 1e-1
grad_clip      = 1.0
