# Experiment 09: Finetune pretrained GPT-2, GQA with 4 KV heads, FineWeb-Edu streaming
# No local train.bin needed — streams directly from HuggingFace.
# Only val.bin required (~8MB): python3 data/fineweb_edu/prepare.py --val_only
#
# Run: python3 train_fsdp.py config/experiments/exp09_gpt2_gqa4_fineweb_stream.py

import os as _os
exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp09_gpt2_gqa4_fineweb_stream'
dataset        = 'hf:HuggingFaceFW/fineweb-edu'  # streams from HF, no local train.bin needed
n_kv_head      = 4              # GQA: 4 KV heads shared across 12 query heads (3 queries per KV)
use_rope       = False          # use absolute PE to match GPT-2 pretrained weights
wandb_run_name = 'exp09-gpt2-gqa4-fineweb-stream'

# GPT-2 architecture
n_layer = 12
n_head  = 12
n_embd  = 768
block_size = 1024
batch_size = 8
gradient_accumulation_steps = 4
dropout = 0.0  # no dropout when finetuning from pretrained

# Auto-detect: resume from checkpoint if one exists, else start fresh from GPT-2
_ckpt = _os.path.join(out_dir, 'ckpt.pt')
if _os.path.exists(_ckpt):
    # Phase 2: fine-tuning with low LR and simple cosine decay
    init_from         = 'resume'
    max_iters         = 20000
    eval_interval     = 500
    warmup_iters      = 0
    learning_rate     = 5e-5
    min_lr            = 5e-6
    lr_scheduler      = 'cosine'
    lr_decay_iters    = 20000   # will decay from current iter_num to max_iters
else:
    # Phase 1: GQA recovery with aggressive LR and cosine restarts
    init_from         = 'gpt2_to_gqa'
    max_iters         = 20000
    eval_interval     = 500
    warmup_iters      = 100
    learning_rate     = 2e-4
    min_lr            = 3e-6
    lr_scheduler      = 'cosine_restarts'
    lr_restart_period = 1000
    lr_restart_mult   = 2
    lr_restart_decay  = 0.75
    lr_decay_iters    = 20000
