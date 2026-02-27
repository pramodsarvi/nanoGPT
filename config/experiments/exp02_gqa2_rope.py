# Experiment 02: GQA (2 KV heads, 3 groups) + RoPE
# 6 query heads / 2 KV heads = 3 queries share each KV head.
#
# Run: ~/venv/bin/python train_fsdp.py config/experiments/exp02_gqa2_rope.py

exec(open('config/experiments/base.py').read())

out_dir       = 'out_experiments/exp02_gqa2_rope'
n_kv_head     = 2
rope_base     = 10000
wandb_run_name = 'exp02-gqa2-rope10k'
