# Experiment 20: ~350M parameter Kannada language model pretrain from scratch
#
#   Architecture:  GQA (n_kv_head=4 of 8) + SwiGLU + RoPE base=10000
#   Tokenizer:     Custom Kannada BPE (16K vocab, trained by prepare_tokenizer.py)
#   Dataset:       Kannada Wikipedia + Sangraha (~1.5B tokens)
#   FSDP:          FULL_SHARD (works on single GPU too — set nproc_per_node=1)
#
# Why 350M for Kannada?
#   Kannada has ~1.5B tokens available (Wikipedia + Sangraha).
#   Chinchilla: optimal model size = tokens / 20 → 1.5B/20 = 75M minimum.
#   350M is a good sweet spot: larger than Chinchilla-minimum, well-trained,
#   fits comfortably on a single RTX 4070 8GB for inference.
#
# VRAM estimates:
#   Single RTX 4070 8GB (nproc=1):
#     Model bf16: 700MB,  Adam fp32: 1.4GB,  Activations (batch=4, block=1024): ~3GB
#     Peak: ~5.5GB → fits in 8GB
#
#   4× RTX 4070 8GB (nproc=4):
#     FSDP shards weights → 175MB/GPU,  activations: ~3GB/GPU
#     batch=4, accum=16 → 4×4×4×1024 = 65,536 tokens/iter
#
#   8× RTX 5060 Ti 16GB (nproc=8):
#     batch=8, accum=32 → 8×4×8×2048 = 524,288 tokens/iter — fast
#
# Token budget:
#   1.5B tokens / 65,536 tokens per iter (4-GPU) = ~22,900 iters   (~4.5 hours)
#   1.5B tokens / 131,072 tokens per iter (8-GPU) = ~11,450 iters  (~2.5 hours)
#
# Prerequisites:
#   1. python data/kannada/prepare_tokenizer.py    # ~30-60 min
#   2. python data/kannada/prepare_data.py         # ~1-2 hours
#
# Run (single GPU):
#   python train_fsdp.py config/experiments/exp20_kannada_350m.py
#
# Run (4 GPU):
#   torchrun --standalone --nproc_per_node=4 train_fsdp.py \
#       config/experiments/exp20_kannada_350m.py
#
# Run (8× RTX 5060 Ti):
#   torchrun --standalone --nproc_per_node=8 train_fsdp.py \
#       config/experiments/exp20_kannada_350m.py \
#       --batch_size=8 --block_size=2048 --gradient_accumulation_steps=32

import os as _os

out_dir   = 'out_experiments/exp20_kannada_350m'
init_from = 'resume' if _os.path.exists(out_dir + '/ckpt.pt') else 'scratch'

# ── Dataset ───────────────────────────────────────────────────────────────────
# Local .bin files prepared by data/kannada/prepare_data.py
# train_fsdp.py resolves this to data/kannada/train.bin + val.bin + meta.pkl
dataset = 'kannada'

# ── Model ─────────────────────────────────────────────────────────────────────
# ~350M params:
#   n_layer=24, n_head=16, n_embd=1024
#   params ≈ 12 × n_layer × n_embd² = 12 × 24 × 1024² = 301M (+ embeddings ~16M)
#   total ≈ 317M params — call it "350M class"
n_layer    = 24
n_head     = 16
n_kv_head  = 4          # GQA 4:1 ratio (16 query heads, 4 kv heads)
n_embd     = 1024
block_size = 1024       # 1024 for single GPU fit; increase to 2048 on multi-GPU
bias       = False
dropout    = 0.0

use_rope   = True
rope_base  = 10000      # standard RoPE — sufficient for 1024 context
use_swiglu = True
gradient_checkpointing = False  # not needed at 350M on single GPU
                                 # set True if OOM on smaller VRAM

# vocab_size is read automatically from data/kannada/meta.pkl (16000)

# ── Batch / gradient accumulation ─────────────────────────────────────────────
# Single GPU (RTX 4070 8GB):
#   effective_batch = 4 × 16 × 1 × 1024 = 65,536 tokens/iter
batch_size                  = 4
gradient_accumulation_steps = 16

# ── Optimizer ─────────────────────────────────────────────────────────────────
learning_rate = 3e-4
min_lr        = 3e-5
beta1         = 0.9
beta2         = 0.95
weight_decay  = 1e-1
grad_clip     = 1.0

# ── LR schedule ───────────────────────────────────────────────────────────────
# 1.5B tokens / 65,536 tokens/iter ≈ 22,900 iters (single GPU)
# Set conservatively — auto-resume will continue if interrupted
decay_lr       = True
lr_scheduler   = 'cosine'
max_iters      = 23000
warmup_iters   = 500
lr_decay_iters = 23000

# ── Eval / checkpointing ──────────────────────────────────────────────────────
eval_interval          = 500
eval_iters             = 50
always_save_checkpoint = True
log_interval           = 10

# ── System ────────────────────────────────────────────────────────────────────
compile        = True
wandb_log      = False
wandb_project  = 'nanogpt'
wandb_run_name = 'exp20-kannada-350m-gqa4-swiglu-rope'
