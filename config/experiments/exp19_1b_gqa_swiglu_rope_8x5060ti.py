# Experiment 19: ~1.2B parameter pretrain from scratch
#   Architecture:  GQA (n_kv_head=8 of 16) + SwiGLU + RoPE  — identical to exp17
#   Tokenizer:     BPE (tiktoken GPT-2, vocab=50304)
#   Dataset:       FineWeb-Edu 350BT (HF streaming)
#   FSDP:          FULL_SHARD across 8 GPUs
#   Target GPUs:   8× RTX 5060 Ti 16GB
#
# Architecture is identical to exp17 — only the batch/accum numbers change
# to fit 16GB VRAM instead of 80GB. Same model, same dataset, same token budget.
#
# VRAM estimate per GPU (FSDP FULL_SHARD, bf16, 8× 16GB):
#   Model params bf16:       2.4GB total  →  300MB/GPU  (sharded ÷8)
#   Optimizer (fp32 Adam):   4.8GB total  →  600MB/GPU  (sharded ÷8)
#   Activations (batch=4, block=4096):     ~7.0GB/GPU
#   Buffers + fragmentation:               ~1.2GB/GPU
#   Peak estimate:                         ~9.1GB/GPU  →  fits in 16GB
#
# Chinchilla-optimal token budget for 1.2B params ≈ 24B tokens
#   effective_batch = batch_size × (grad_accum ÷ world_size) × world_size × block_size
#                   = 4 × 8 × 8 × 4096 = 2,097,152 ≈ 2M tokens/iter
#   24B / 2M = 12,000 iters
#
# To run:
#   torchrun --standalone --nproc_per_node=8 train_fsdp.py \
#       config/experiments/exp19_1b_gqa_swiglu_rope_8x5060ti.py
#
# For a quick sanity-check (200 iters, ~15 min):
#   torchrun --standalone --nproc_per_node=8 train_fsdp.py \
#       config/experiments/exp19_1b_gqa_swiglu_rope_8x5060ti.py \
#       --max_iters=200 --eval_interval=50

import os as _os

out_dir   = 'out_experiments/exp19_1b_gqa_swiglu_rope_8x5060ti'
init_from = 'resume' if _os.path.exists(out_dir + '/ckpt.pt') else 'scratch'

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = 'hf:HuggingFaceFW/fineweb-edu@sample-350BT'

# RTX 5060 Ti has fast PCIe but less memory bandwidth than A100;
# moderate prefetch to avoid host RAM pressure across 8 processes
hf_prefetch_batches  = 64
hf_tokenizer_threads = 4

# ── Model — identical to exp17 ────────────────────────────────────────────────
n_layer    = 24
n_head     = 16
n_kv_head  = 8          # GQA 2:1 ratio
n_embd     = 2048
block_size = 4096
bias       = False
dropout    = 0.0

use_rope   = True
rope_base  = 500000     # LLaMA-3-style long-context base
use_swiglu = True
gradient_checkpointing = True   # essential on 16GB — saves ~30% activation memory
                                 # reduces peak from ~9GB to ~6.5GB/GPU

# ── Batch / gradient accumulation ─────────────────────────────────────────────
# 16GB VRAM constrains micro-batch to 4 at block=4096 (even with checkpointing).
# 8 GPUs × larger accum restores the Chinchilla-optimal effective batch.
#
# effective_batch = 4 × 8 × 8 × 4096 = 2,097,152 tokens/iter
batch_size                  = 4    # micro-batch per GPU — tight at 16GB, block=4096
gradient_accumulation_steps = 64   # 64 ÷ 8 GPUs = 8 accum steps per GPU
                                    # → 4 × 8 × 8 × 4096 = 2,097,152 tokens/iter

# ── Optimizer — identical to exp17 ───────────────────────────────────────────
learning_rate = 3e-4
min_lr        = 3e-5
beta1         = 0.9
beta2         = 0.95
weight_decay  = 1e-1
grad_clip     = 1.0

# ── LR schedule — identical to exp17 ─────────────────────────────────────────
decay_lr       = True
lr_scheduler   = 'cosine'
max_iters      = 12000      # 12000 × 2,097,152 = 25.2B tokens
warmup_iters   = 400
lr_decay_iters = 12000

# ── Eval / checkpointing ──────────────────────────────────────────────────────
eval_interval          = 250
eval_iters             = 20
always_save_checkpoint = True
log_interval           = 1

# ── System ────────────────────────────────────────────────────────────────────
compile        = True
wandb_log      = False
wandb_project  = 'nanogpt'
wandb_run_name = 'exp19-1b-gqa8-swiglu-rope500k-4096ctx-8x5060ti'
