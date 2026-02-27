# Experiment 03: GQA (3 KV heads, 2 groups) + RoPE
# 6 query heads / 3 KV heads = 2 queries share each KV head.
# More KV heads than exp02 — closer to MHA, less aggressive compression.
#
# Run: ~/venv/bin/python train_fsdp.py config/experiments/exp03_gqa3_rope.py

exec(open('config/experiments/base.py').read())

out_dir       = 'out_experiments/exp03_gqa3_rope'
n_kv_head     = 3
rope_base     = 10000
wandb_run_name = 'exp03-gqa3-rope10k'
