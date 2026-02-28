# Shared base settings for all Shakespeare char comparison experiments.
# Each experiment config imports this via exec() by the configurator,
# then overrides only what differs.
#
# DO NOT run this file directly — use the experiment-specific configs below.

out_dir = 'out_experiments/base'   # overridden per experiment
dataset = 'shakespeare_char'
eval_interval = 500
log_interval  = 10
eval_iters    = 100
eval_only     = False
always_save_checkpoint = False

init_from = 'scratch'

gradient_accumulation_steps = 8
batch_size  = 8
block_size  = 512

# Model dims — same across all experiments for fair comparison
n_layer = 6
n_head  = 6
n_embd  = 384
dropout = 0.1
bias    = False

# Optimizer
learning_rate = 6e-4
max_iters     = 20000
weight_decay  = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
decay_lr      = True
warmup_iters  = 500
lr_decay_iters = 20000
min_lr        = 6e-5

# System
device  = 'cuda'
compile = False
wandb_log     = False
wandb_project = 'nanogpt-experiments'
