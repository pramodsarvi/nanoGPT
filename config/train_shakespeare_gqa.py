# Train a small GQA model on shakespeare_char to validate the implementation.
# Single GPU, fast iteration — not meant for quality, just correctness.
#
# Run:
#   ~/venv/bin/python train_fsdp.py config/train_shakespeare_gqa.py

out_dir = 'out_shakespeare_gqa'
eval_interval = 250
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = False

init_from = 'scratch'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 256

# Small model — fits easily in 8GB
n_layer = 6
n_head = 6
n_kv_head = 2   # GQA: 3 query heads share each KV head (6 / 2 = 3 groups)
n_embd = 384
dropout = 0.1
bias = False

learning_rate = 1e-3
max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 1e-4

# Single GPU — no torchrun needed
backend = 'nccl'
device = 'cuda'
compile = False   # keep off for quick debug; enable once basics work

wandb_log = False
