# Experiment 08: GQA verification on FineWeb-Edu (streaming)
# Goal: verify GQA implementation trains correctly on a real dataset
# - Check loss decreases consistently
# - Check train/val gap stays reasonable (no wild overfitting)
# - Check checkpoint save/resume works
#
# Run: python3 train_fsdp.py config/experiments/exp08_gqa_fineweb_verify.py

exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp08_gqa_fineweb_verify'
init_from      = 'gpt2_to_gqa'
dataset        = 'hf:HuggingFaceFW/fineweb-edu'  # streams from HF, no local train.bin needed
n_kv_head      = 2
wandb_run_name = 'exp08-gqa-fineweb-verify'

# GPT-2 scale
n_layer = 12
n_head  = 12
n_embd  = 768
block_size = 1024
batch_size = 8
gradient_accumulation_steps = 4
dropout = 0.0   # no dropout when finetuning from pretrained (standard practice)

# Short run — just verify convergence
max_iters      = 1000
eval_interval  = 200
warmup_iters   = 50
learning_rate  = 3e-5
min_lr         = 3e-6
lr_decay_iters = 1000
