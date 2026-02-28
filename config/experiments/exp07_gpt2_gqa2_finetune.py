# Experiment 07: Load pretrained GPT-2, convert to GQA (n_kv_head=2), finetune
#
# Run: python3 train_fsdp.py config/experiments/exp07_gpt2_gqa2_finetune.py

exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp07_gpt2_gqa2_finetune'
init_from      = 'gpt2_to_gqa'  # load pretrained GPT-2, convert MHA → GQA
n_kv_head      = 2              # GQA: 2 KV heads shared across 12 query heads
wandb_run_name = 'exp07-gpt2-gqa2-finetune'

# GPT-2 architecture (overrides base.py model dims)
n_layer = 12
n_head  = 12
n_embd  = 768
block_size = 1024
batch_size = 8
gradient_accumulation_steps = 4

# Finetune with lower LR
max_iters      = 5000
learning_rate  = 3e-5
min_lr         = 3e-6
warmup_iters   = 100
lr_decay_iters = 5000


init_from='resume'

