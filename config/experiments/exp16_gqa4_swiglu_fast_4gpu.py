# Experiment 16: Fast GPT-2 Small pretrain — GQA + SwiGLU + RoPE on 4× H100
# Optimized for wall-clock speed and fast convergence:
#   - Local memmap data (no HF streaming overhead)
#   - Large batch size to saturate H100 compute
#   - Chinchilla-optimal token budget (~2.6B tokens for 124M model)
#   - Minimal grad accumulation overhead (1 step per GPU)
#   - Lean eval settings
#
# Data prep (run once, ~5 min):
#   python3 data/fineweb_edu/prepare.py --max_tokens 10_000_000_000
#
# Run on 4× H100:
#   torchrun --standalone --nproc_per_node=4 train_fsdp.py config/experiments/exp16_gqa4_swiglu_fast_4gpu.py
#
# Expected: ~40-60 min on 4× H100 80GB

import os as _os
exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp16_gqa4_swiglu_fast_4gpu'
dataset        = 'fineweb_edu'              # local memmap — run prepare.py first!
init_from      = 'resume' if _os.path.exists('out_experiments/exp16_gqa4_swiglu_fast_4gpu/ckpt.pt') else 'scratch'
n_kv_head      = 4
use_rope       = True
use_swiglu     = True
wandb_run_name = 'exp16-gqa4-swiglu-fast-4gpu'

# GPT-2 Small architecture
n_layer    = 12
n_head     = 12
n_embd     = 768
block_size = 1024
bias       = False
dropout    = 0.0
compile    = True

# Throughput — maximize GPU utilization
batch_size = 128                  # large micro-batch; fits H100 80GB with FSDP FULL_SHARD
gradient_accumulation_steps = 4   # ÷ 4 GPUs = 1 accum step each → zero overhead
                                  # effective batch = 128 × 1024 × 4 = 524,288 tokens/iter

# Chinchilla-optimal training duration
# 124M params × ~20 tokens/param ≈ 2.5B tokens
# 2.6B / 524K ≈ 5000 iters
max_iters      = 5000
lr_decay_iters = 5000
warmup_iters   = 400              # 8% warmup — fast ramp

# LR — aggressive but stable with large batch + warmup
learning_rate  = 1e-3
min_lr         = 1e-4             # 10× decay
lr_scheduler   = 'cosine'
weight_decay   = 1e-1
grad_clip      = 1.0

# Lean eval — don't waste compute
eval_interval  = 250              # eval 20 times over the run
eval_iters     = 20               # 20 batches per split is enough at batch_size=128
always_save_checkpoint = True     # save every eval (cheap insurance)
