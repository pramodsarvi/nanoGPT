# Experiment 06: Finetune pretrained GPT-2 (MHA, absolute PE) on shakespeare_char
#
# Run: python3 train_fsdp.py config/experiments/exp06_gpt2_mha_finetune.py

exec(open('config/experiments/base.py').read())

out_dir        = 'out_experiments/exp06_gpt2_mha_finetune'
init_from      = 'gpt2'         # load pretrained GPT-2 124M from HuggingFace
n_kv_head      = 0              # MHA — keep as-is
wandb_run_name = 'exp06-gpt2-mha-finetune'

# GPT-2 architecture (overrides base.py model dims)
n_layer = 12
n_head  = 12
n_embd  = 768
block_size = 1024

# Finetune with lower LR
learning_rate = 3e-5
min_lr        = 3e-6
warmup_iters  = 100
lr_decay_iters = 20000
