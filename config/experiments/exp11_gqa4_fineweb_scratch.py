# Experiment 11: GPT-2 Small from scratch with GQA (n_kv_head=4) on FineWeb-Edu
# Clean from-scratch training — no pretrained weights, no conversion mess.
# Pair with exp12 (MHA baseline) for a direct GQA vs MHA comparison.
#
# Data prep (run once on A100, ~5 min for 1BT):
#   python data/fineweb_edu/prepare.py --max_tokens 1_000_000_000
#
# Run:
#   python train_fsdp.py config/experiments/exp11_gqa4_fineweb_scratch.py

import os as _os
exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp11_gqa4_fineweb_scratch'
dataset        = 'hf:HuggingFaceFW/fineweb-edu'
init_from      = 'resume' if _os.path.exists('out_experiments/exp11_gqa4_fineweb_scratch/ckpt.pt') else 'scratch'
n_kv_head      = 4              # GQA: 4 KV heads, 3 queries per KV head
use_rope       = True           # RoPE — better for from-scratch training
wandb_run_name = 'exp11-gqa4-fineweb-scratch'

# GPT-2 Small architecture
n_layer    = 12
n_head     = 12
n_embd     = 768
block_size = 1024
batch_size = 48
gradient_accumulation_steps = 4   # effective batch = 192
dropout    = 0.0
compile    = True

# Single-cycle warmup + cosine decay
max_iters      = 50000
eval_interval  = 500
warmup_iters   = 500
learning_rate  = 6e-4      # standard from-scratch LR for GPT-2 Small
min_lr         = 6e-5      # 10x decay
lr_scheduler   = 'cosine'
lr_decay_iters = 50000
weight_decay   = 1e-1
grad_clip      = 1.0
