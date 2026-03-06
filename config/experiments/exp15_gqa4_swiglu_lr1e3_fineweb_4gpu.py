# Experiment 15: GPT-2 Small from scratch, GQA (n_kv_head=4) + SwiGLU + lr=1e-3
# Same as exp14 but tuned for 2× H100 with FSDP.
# gradient_accumulation_steps must be divisible by world_size (2).
#
# Run on 2× H100 with FSDP:
#   torchrun --standalone --nproc_per_node=2 train_fsdp.py config/experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu.py

import os as _os
exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu'
dataset        = 'hf:HuggingFaceFW/fineweb-edu'
init_from      = 'resume' if _os.path.exists('out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu/ckpt.pt') else 'scratch'
n_kv_head      = 4
use_rope       = True
use_swiglu     = True
wandb_run_name = 'exp15-gqa4-swiglu-lr1e3-4gpu'

# GPT-2 Small architecture
n_layer    = 12
n_head     = 12
n_embd     = 768
block_size = 1024
batch_size = 80                   # Increased to 80 for better 80GB memory utilization
gradient_accumulation_steps = 10  # 10 * 80 * 1024 = 819,200 (Roughly same effective batch)
                                  # divisible by 2 GPUs → 5 accum steps per GPU
hf_prefetch_batches  = 128         # Lowered from 256 to reduce CPU/pinned RAM usage
hf_tokenizer_threads = 8          # Keep at 1 to avoid redundant dataloader RAM usage
dropout    = 0.0
compile    = True

# Higher LR with proportionally higher min_lr (keep 10x ratio)
# Total tokens ≈ 786K × 11250 ≈ 8.85B (matches exp14's 590K × 15000 ≈ 8.85B)
max_iters      = 20000
eval_interval  = 375              # ~same number of evals as exp14 (15000/500 = 30, 11250/375 = 30)
warmup_iters   = 750              # same fraction as exp14 (1000/15000 ≈ 750/11250 ≈ 6.7%)
learning_rate  = 1e-3
min_lr         = 1e-4             # 10x decay
lr_scheduler   = 'cosine'
lr_decay_iters = 20000
weight_decay   = 1e-1
grad_clip      = 1.0
