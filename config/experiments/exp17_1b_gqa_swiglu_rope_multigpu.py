# Experiment 17: ~1.2B parameter pretrain from scratch
#   Architecture:  GQA (n_kv_head=8 of 16) + SwiGLU + RoPE
#   Tokenizer:     BPE (tiktoken GPT-2, vocab=50304)
#   Dataset:       FineWeb-Edu 350BT (HF streaming — no local disk needed)
#   FSDP:          FULL_SHARD across all GPUs (real parameter + grad + optimizer sharding)
#   Target GPUs:   4× A100 80GB
#
# Parameter estimate (approximate):
#   n_layer=24, n_head=16, n_embd=2048, n_kv_head=8, SwiGLU hidden=5504
#   Per block: q(4.2M) + kv(4.2M) + c_proj(4.2M) + gate(11.3M) + up(11.3M) + down(11.3M) ≈ 46M
#   24 blocks:  ~1.1B  +  vocab embedding 50304×2048 ≈ 103M  →  ~1.2B total params
#
# VRAM breakdown per GPU (4× 80GB, FSDP FULL_SHARD, bf16):
#   Model params bf16:       2.4 GB total  → 0.6 GB/GPU (sharded)
#   Optimizer (fp32 Adam):   4.8 GB total  → 1.2 GB/GPU (sharded)
#   Activations (batch=16):  ~28 GB/GPU (block=4096 doubles vs block=2048, so batch halved)
#   Buffers + fragmentation: ~4 GB/GPU
#   Peak estimate:           ~34 GB/GPU  →  comfortable on 80GB
#
# Chinchilla-optimal token budget for 1.2B params ≈ 24B tokens
#   effective_batch = batch_size × (grad_accum ÷ world_size) × world_size × block_size
#                   = 16 × 8 × 4 × 4096 = 2,097,152 ≈ 2M tokens/iter  (same as before)
#   24B / 2M = 12,000 iters  (no padding needed)
#
# To run on 4× A100 80GB:
#   torchrun --standalone --nproc_per_node=4 train_fsdp.py \
#       config/experiments/exp17_1b_gqa_swiglu_rope_multigpu.py
#
# For a quick sanity-check (500 iters, ~5 min):
#   torchrun --standalone --nproc_per_node=4 train_fsdp.py \
#       config/experiments/exp17_1b_gqa_swiglu_rope_multigpu.py \
#       --max_iters=500 --eval_interval=100

import os as _os

out_dir   = 'out_experiments/exp17_1b_gqa_swiglu_rope_multigpu'
init_from = 'resume' if _os.path.exists(out_dir + '/ckpt.pt') else 'scratch'

# ── Dataset ──────────────────────────────────────────────────────────────────
# FineWeb-Edu 350BT (high-quality educational web text, ~350B tokens).
# Uses HF streaming — no local data prep needed.
# Switch to sample-10BT for a quicker run:
#   dataset = 'hf:HuggingFaceFW/fineweb-edu'
dataset = 'hf:HuggingFaceFW/fineweb-edu@sample-350BT'

# HF prefetch — A100s are fast; keep the pipeline well ahead
hf_prefetch_batches  = 128
hf_tokenizer_threads = 8

# ── Model ─────────────────────────────────────────────────────────────────────
# LLaMA-1-1.3B-style: 24 layers, 16 heads, 2048 embd, 8 KV heads, 2048 ctx
n_layer    = 24
n_head     = 16
n_kv_head  = 8          # GQA ratio 2:1 (each KV head serves 2 query heads)
n_embd     = 2048
block_size = 4096
bias       = False
dropout    = 0.0

use_rope       = True
rope_base      = 500000  # LLaMA-3-style long-context base (good for block_size=2048+)
use_swiglu     = True
gradient_checkpointing = False  # not needed on 80GB A100 at batch=32; off = ~15% faster

# vocab_size is auto-detected from tiktoken BPE (50304)

# ── Batch / gradient accumulation ─────────────────────────────────────────────
# 4× A100 80GB — block_size=4096 doubles activation memory, so batch_size halved vs 2048.
# Rule: gradient_accumulation_steps must be divisible by world_size (4).
#
# effective_batch = batch_size × (grad_accum ÷ world_size) × world_size × block_size
#                 = 16 × 8 × 4 × 4096 = 2,097,152 tokens/iter  (Chinchilla target)
batch_size                  = 16   # micro-batch per GPU — halved to fit 4096 ctx in 80GB
gradient_accumulation_steps = 32   # 32 ÷ 4 GPUs = 8 accum steps per GPU
                                    # → 16 × 8 × 4 × 4096 = 2,097,152 tokens/iter

# ── Optimizer ─────────────────────────────────────────────────────────────────
# Scaled LR following μP / Chinchilla conventions for 1B-scale models
learning_rate = 3e-4        # lower than small models — standard for 1B+ scale
min_lr        = 3e-5        # 10× decay
beta1         = 0.9
beta2         = 0.95
weight_decay  = 1e-1
grad_clip     = 1.0

# ── LR schedule ───────────────────────────────────────────────────────────────
decay_lr       = True
lr_scheduler   = 'cosine'
max_iters      = 12000      # 12000 × 2,097,152 = 25.2B tokens (~Chinchilla-optimal for 1.2B)
warmup_iters   = 400        # ~800M token warmup
lr_decay_iters = 12000

# ── Eval / checkpointing ──────────────────────────────────────────────────────
eval_interval          = 250    # eval ~48 times over the run
eval_iters             = 20     # 20 batches per split (fast eval at 1B scale)
always_save_checkpoint = True
log_interval           = 1

# ── System ────────────────────────────────────────────────────────────────────
compile        = True           # torch.compile speeds up training ~10-20%
wandb_log      = False          # set True to track on Weights & Biases
wandb_project  = 'nanogpt'
wandb_run_name = 'exp17-1b-gqa8-swiglu-rope500k-4096ctx-4xA100'
