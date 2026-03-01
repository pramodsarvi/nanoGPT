# Experiment 13: GPT-2 Small from scratch with GQA (n_kv_head=4) + SwiGLU on FineWeb-Edu
# Identical to exp11 except MLP activation: SwiGLU instead of GELU.
# SwiGLU is used by LLaMA, Mistral, PaLM — expected ~5-10% better loss for same compute.
#
# Run:
#   python train_fsdp.py config/experiments/exp13_gqa4_swiglu_fineweb_scratch.py

import os as _os
exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp13_gqa4_swiglu_fineweb_scratch'
dataset        = 'hf:HuggingFaceFW/fineweb-edu'
init_from      = 'resume' if _os.path.exists('out_experiments/exp13_gqa4_swiglu_fineweb_scratch/ckpt.pt') else 'scratch'
n_kv_head      = 4              # GQA: 4 KV heads, 3 queries per KV head
use_rope       = True
use_swiglu     = True           # SwiGLU MLP — LLaMA-style gated activation
wandb_run_name = 'exp13-gqa4-swiglu-fineweb-scratch'

# GPT-2 Small architecture — identical to exp11
n_layer    = 12
n_head     = 12
n_embd     = 768
block_size = 1024
batch_size = 48
gradient_accumulation_steps = 4   # effective batch = 192
dropout    = 0.0
compile    = True

# Identical schedule to exp11
max_iters      = 20000
eval_interval  = 500
warmup_iters   = 500
learning_rate  = 6e-4
min_lr         = 6e-5
lr_scheduler   = 'cosine'
lr_decay_iters = 20000
weight_decay   = 1e-1
grad_clip      = 1.0
