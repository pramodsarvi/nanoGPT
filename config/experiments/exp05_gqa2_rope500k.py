# Experiment 05: GQA (2 KV heads) + RoPE with LLaMA3-style high rope_base
# rope_base=500000 improves long-context extrapolation.
# On short sequences (block_size=256) effect is minimal but worth tracking.
#
# Run: ~/venv/bin/python train_fsdp.py config/experiments/exp05_gqa2_rope500k.py

exec(open('config/experiments/base.py').read())

out_dir       = 'out_experiments/exp05_gqa2_rope500k'
n_kv_head     = 2
rope_base     = 500000
wandb_run_name = 'exp05-gqa2-rope500k'
