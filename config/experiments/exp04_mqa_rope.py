# Experiment 04: MQA (Multi-Query Attention, 1 KV head) + RoPE
# Extreme case: all 6 query heads share a single KV head.
# Maximum KV cache compression, lowest memory, may hurt quality.
#
# Run: ~/venv/bin/python train_fsdp.py config/experiments/exp04_mqa_rope.py

exec(open('config/experiments/base.py').read())

out_dir       = 'out_experiments/exp04_mqa_rope'
n_kv_head     = 1
rope_base     = 10000
wandb_run_name = 'exp04-mqa-rope10k'
