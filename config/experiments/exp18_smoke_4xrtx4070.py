# Experiment 18: Smoke test — verify FSDP training loop on 4× RTX 4070 12GB
#
#   This is NOT a real training run. The goal is to verify:
#     - FSDP wrapping, sharding, and grad sync work correctly
#     - HF streaming data pipeline feeds all 4 ranks without stalling
#     - Checkpointing (save + auto-resume) works end-to-end
#     - Loss decreases — i.e. the model is actually learning
#
#   Architecture is ~125M params (GPT-2 Small scale) — same code path as
#   exp17 (GQA + SwiGLU + RoPE) so any bug here will surface at 1B scale too.
#   With 12GB VRAM there is enough headroom to use a real batch size.
#
# VRAM estimate per GPU (FSDP FULL_SHARD, bf16, 4× 12GB):
#   Model params bf16:      250MB total  → 63MB/GPU (sharded)
#   Optimizer (fp32 Adam):  500MB total  → 125MB/GPU (sharded)
#   Activations (batch=8):  ~3.5GB/GPU
#   Buffers + fragmentation: ~0.8GB/GPU
#   Peak estimate:          ~4.5GB/GPU  →  comfortable on 12GB
#
# Effective batch:
#   batch_size × (grad_accum ÷ world_size) × world_size × block_size
#   = 8 × 4 × 4 × 1024 = 131,072 tokens/iter
#
# To run:
#   torchrun --standalone --nproc_per_node=4 train_fsdp.py \
#       config/experiments/exp18_smoke_4xrtx4070.py
#
# Expected output:
#   - iter 0:   loss ~10.8  (random init)
#   - iter 50:  loss ~6-7   (visibly dropping)
#   - iter 200: loss ~4-5   (clearly learning)
#   Checkpoint saved at iter 0, 50, 100, 150, 200.
#   Re-run the same command to verify auto-resume works.

import os as _os

out_dir   = 'out_experiments/exp18_smoke_4xrtx4070'
init_from = 'resume' if _os.path.exists(out_dir + '/ckpt.pt') else 'scratch'

# ── Dataset ───────────────────────────────────────────────────────────────────
# Use sample-10BT (smaller HF config) — less network overhead for a quick test.
# Switch to shakespeare_char for a fully offline test:
#   dataset = 'shakespeare_char'  (run python data/shakespeare_char/prepare.py first)
dataset = 'hf:HuggingFaceFW/fineweb-edu'   # defaults to sample-10BT

# Light prefetch — RTX 4070 is slower than A100, less need to buffer ahead
hf_prefetch_batches  = 16
hf_tokenizer_threads = 2

# ── Model — GPT-2 Small scale, same architecture family as exp17 ──────────────
# ~125M params: real GPT-2 Small size with GQA+SwiGLU+RoPE — same code path
# as the 1B run so any bug here will also surface at 1B scale.
n_layer   = 12
n_head    = 12
n_kv_head = 4       # GQA 3:1 ratio (same code path as exp17's 2:1)
n_embd    = 768
block_size = 1024   # full GPT-2 context — fits in 12GB at batch=8
bias      = False
dropout   = 0.0

use_rope       = True
rope_base      = 500000   # same as exp17 — tests RoPE at this base
use_swiglu     = True     # same as exp17 — tests SwiGLU path
gradient_checkpointing = False  # not needed at this tiny scale

# ── Batch ─────────────────────────────────────────────────────────────────────
# 12GB gives comfortable headroom at batch=8, block=1024.
# effective_batch = 8 × 4 × 4 × 1024 = 131,072 tokens/iter
batch_size                  = 8    # micro-batch per GPU
gradient_accumulation_steps = 16   # 16 ÷ 4 GPUs = 4 accum steps per GPU

# ── Optimizer ─────────────────────────────────────────────────────────────────
learning_rate = 3e-4
min_lr        = 3e-5
beta1         = 0.9
beta2         = 0.95
weight_decay  = 1e-1
grad_clip     = 1.0

# ── Schedule — short run, just enough to confirm loss drops ───────────────────
decay_lr       = True
lr_scheduler   = 'cosine'
max_iters      = 200       # ~5 min on 4× RTX 4070
warmup_iters   = 20
lr_decay_iters = 200

# ── Eval / checkpointing ──────────────────────────────────────────────────────
eval_interval          = 50    # eval 4 times over the run
eval_iters             = 10
always_save_checkpoint = True
log_interval           = 10

# ── System ────────────────────────────────────────────────────────────────────
compile   = False   # off — torch.compile has a long first-step cost; not worth it for 200 iters
wandb_log = False
wandb_project  = 'nanogpt'
wandb_run_name = 'exp18-smoke-gqa4-swiglu-rope500k-4xRTX4070-12GB'
