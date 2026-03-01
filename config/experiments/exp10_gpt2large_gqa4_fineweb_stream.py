# Experiment 10: Finetune pretrained GPT-2 Small, GQA with 4 KV heads, FineWeb-Edu streaming
# GPT-2 Small: n_layer=12, n_head=12, n_embd=768 (117M params)
# n_kv_head=4: 4 KV heads shared across 12 query heads (3 queries per KV)
# No local train.bin needed — streams directly from HuggingFace.
# Only val.bin required (~8MB): python3 data/fineweb_edu/prepare.py --val_only
#
# Run: python3 train_fsdp.py config/experiments/exp10_gpt2large_gqa4_fineweb_stream.py

import os as _os
exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp10_gpt2large_gqa4_fineweb_stream'
dataset        = 'hf:HuggingFaceFW/fineweb-edu'
n_kv_head      = 4
use_rope       = False
wandb_run_name = 'exp10-gpt2small-gqa4-fineweb-stream'

# GPT-2 Small architecture
n_layer    = 12
n_head     = 12
n_embd     = 768
block_size = 1024
batch_size = 32
gradient_accumulation_steps = 4   # effective batch = 128
dropout    = 0.0
compile    = True

# Auto-detect: resume if checkpoint exists, else start from GPT-2 pretrained
_ckpt = _os.path.join(out_dir, 'ckpt.pt')
init_from = 'resume' if _os.path.exists(_ckpt) else 'gpt2_to_gqa'

# Single-cycle warmup + cosine decay — stable for both fresh start and resume
max_iters      = 20000
eval_interval  = 500
warmup_iters   = 200       # ramp up over first 200 iters
learning_rate  = 1e-4      # peak LR — conservative enough for pretrained weights
min_lr         = 3e-6      # ~30x decay at end
lr_scheduler   = 'cosine'
lr_decay_iters = 20000
