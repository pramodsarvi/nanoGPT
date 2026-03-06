# Experiment 14: GPT-2 Small from scratch, GQA (n_kv_head=4) + SwiGLU + lr=1e-3
# vs exp11: higher LR (1e-3 vs 6e-4) + SwiGLU activation
# vs exp13: same SwiGLU but higher LR
# Run on 2× H100 with FSDP:
#   torchrun --standalone --nproc_per_node=2 train_fsdp.py config/experiments/exp14_gqa4_swiglu_lr1e3_fineweb_scratch.py

import os as _os
exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp14_gqa4_swiglu_lr1e3_fineweb_scratch'
dataset        = 'hf:HuggingFaceFW/fineweb-edu'
init_from      = 'resume' if _os.path.exists('out_experiments/exp14_gqa4_swiglu_lr1e3_fineweb_scratch/ckpt.pt') else 'scratch'
n_kv_head      = 4
use_rope       = True
use_swiglu     = True
wandb_run_name = 'exp14-gqa4-swiglu-lr1e3-fineweb-scratch'

# GPT-2 Small architecture
n_layer    = 12
n_head     = 12
n_embd     = 768
block_size = 1024
batch_size = 96                   # logits (B, T, vocab) = 9.2GB per GPU
gradient_accumulation_steps = 8   # divided by 4 GPUs = 2 per GPU; effective batch = 96×1024×8×4 = 3,145,728 tokens/step
hf_prefetch_batches  = 128        # large prefetch buffer to keep H100s fed
hf_tokenizer_threads = 1          # single thread — multiple threads cause data duplication
dropout    = 0.0
compile    = False

# Higher LR with proportionally higher min_lr (keep 10x ratio)
max_iters      = 15000
eval_interval  = 500
warmup_iters   = 1000             # longer warmup for higher LR — more stable ramp
learning_rate  = 1e-3
min_lr         = 1e-4             # 10x decay
lr_scheduler   = 'cosine'
lr_decay_iters = 15000
weight_decay   = 1e-1
grad_clip      = 1.0
