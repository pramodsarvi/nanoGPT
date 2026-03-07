"""
Visualise training loss from nanoGPT log files.

Usage:
  ~/venv/bin/python plot_loss.py \
      --log_dir=/home/pramod/Downloads/nanoGPT-checkpoints/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu

  # Compare multiple runs side by side
  ~/venv/bin/python plot_loss.py \
      --log_dir=out_experiments/exp14_gqa4_swiglu_lr1e3_fineweb_scratch \
      --log_dir=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu

  # Save to file instead of displaying
  ~/venv/bin/python plot_loss.py \
      --log_dir=/home/pramod/Downloads/nanoGPT-checkpoints/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu \
      --save=loss_plot.png
"""
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no Qt/GTK, no segfault
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── arg parsing (compatible with configurator.py style too) ──────────────────
log_dirs = []
save     = 'loss_plot.png'   # output file; always saves (Agg backend, no GUI)
smooth   = 50       # smoothing window for step-level train loss (0 = off)
tokens_x = False    # use tokens_seen on x-axis instead of iter

# simple CLI parsing (not using configurator.py since this is a standalone tool)
_args = sys.argv[1:]
_i = 0
while _i < len(_args):
    a = _args[_i]
    if a.startswith('--log_dir='):
        log_dirs.append(a.split('=', 1)[1])
    elif a == '--log_dir' and _i + 1 < len(_args):
        log_dirs.append(_args[_i + 1]); _i += 1
    elif a.startswith('--save='):
        save = a.split('=', 1)[1]
    elif a.startswith('--smooth='):
        smooth = int(a.split('=', 1)[1])
    elif a.startswith('--tokens_x='):
        tokens_x = a.split('=', 1)[1].lower() == 'true'
    _i += 1

if not log_dirs:
    # default to current dir
    log_dirs = ['.']

# ── load & clean logs ─────────────────────────────────────────────────────────

def load_logs(log_dir):
    """
    Load log.csv (eval) and log_steps.csv (per-step train).
    Handles duplicate rows written by multi-GPU ranks — keeps one row per iter
    by averaging across duplicates.
    """
    eval_path  = os.path.join(log_dir, 'log.csv')
    steps_path = os.path.join(log_dir, 'log_steps.csv')

    eval_df  = None
    steps_df = None

    if os.path.exists(eval_path):
        df = pd.read_csv(eval_path)
        # deduplicate: multiple ranks write the same iter — average numeric cols
        num_cols = ['train_loss', 'val_loss', 'lr', 'tokens_seen']
        num_cols = [c for c in num_cols if c in df.columns]
        eval_df = (df.groupby('iter')[num_cols]
                     .mean()
                     .reset_index()
                     .sort_values('iter'))
        print(f"  {log_dir}: eval log — {len(eval_df)} checkpoints, "
              f"iters {eval_df['iter'].min()}–{eval_df['iter'].max()}")

    if os.path.exists(steps_path):
        df = pd.read_csv(steps_path)
        num_cols = ['train_loss', 'lr', 'tokens_seen']
        num_cols = [c for c in num_cols if c in df.columns]
        steps_df = (df.groupby('iter')[num_cols]
                      .mean()
                      .reset_index()
                      .sort_values('iter'))
        print(f"  {log_dir}: steps log — {len(steps_df)} steps, "
              f"iters {steps_df['iter'].min()}–{steps_df['iter'].max()}")

    return eval_df, steps_df


def smooth_series(y, window):
    """Simple moving average."""
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')


def tokens_label(t):
    if t >= 1e9: return f"{t/1e9:.1f}B"
    if t >= 1e6: return f"{t/1e6:.0f}M"
    return f"{t/1e3:.0f}K"


# ── load all runs ─────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'grid.alpha': 0.3,
})

COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4']

runs = []
for idx, log_dir in enumerate(log_dirs):
    print(f"\nLoading: {log_dir}")
    eval_df, steps_df = load_logs(log_dir)
    runs.append({
        'label': os.path.basename(log_dir.rstrip('/')),
        'color': COLORS[idx % len(COLORS)],
        'eval':  eval_df,
        'steps': steps_df,
    })

x_label = 'Tokens seen' if tokens_x else 'Iteration'
x_key   = 'tokens_seen' if tokens_x else 'iter'

def x_vals(df):
    return df[x_key].values if x_key in df.columns else df['iter'].values

def annotate_last(ax, x, y, color):
    ax.annotate(f"{y[-1]:.3f}", xy=(x[-1], y[-1]),
                xytext=(8, 0), textcoords='offset points',
                fontsize=9, color=color, va='center')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Loss only  (train step loss + train/val eval loss)
# ─────────────────────────────────────────────────────────────────────────────
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(14, 5))

for r in runs:
    c, lbl = r['color'], r['label']

    if r['steps'] is not None and 'train_loss' in r['steps'].columns:
        x = x_vals(r['steps'])
        y = r['steps']['train_loss'].values
        ax1a.plot(x, y, alpha=0.15, color=c, linewidth=0.7)
        ax1a.plot(x, smooth_series(y, smooth), color=c, linewidth=1.8,
                  label=f"{lbl} (smooth={smooth})")

    if r['eval'] is not None:
        x = x_vals(r['eval'])
        if 'train_loss' in r['eval'].columns:
            ax1b.plot(x, r['eval']['train_loss'].values,
                      color=c, linewidth=1.5, linestyle='--', alpha=0.7,
                      label=f"{lbl} train")
        if 'val_loss' in r['eval'].columns:
            y = r['eval']['val_loss'].values
            ax1b.plot(x, y, color=c, linewidth=2.2, marker='o', markersize=3,
                      label=f"{lbl} val")
            annotate_last(ax1b, x, y, c)

ax1a.set_title(f'Train Loss — per step  (smooth={smooth})')
ax1a.set_xlabel(x_label); ax1a.set_ylabel('Cross-entropy loss')
ax1a.legend(); ax1a.grid(True)

ax1b.set_title('Train vs Val Loss  (eval checkpoints)')
ax1b.set_xlabel(x_label); ax1b.set_ylabel('Cross-entropy loss')
ax1b.legend(); ax1b.grid(True)

fig1.suptitle('nanoGPT — Loss Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
loss_path = save.replace('.png', '_loss.png') if save.endswith('.png') else save + '_loss.png'
fig1.savefig(loss_path, dpi=150, bbox_inches='tight')
print(f"Saved: {loss_path}")
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Learning rate only
# ─────────────────────────────────────────────────────────────────────────────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

for r in runs:
    c, lbl = r['color'], r['label']
    # use steps (dense) for LR curve shape; eval for milestone markers
    for df, marker, ms in [(r['steps'], None, 0), (r['eval'], 'o', 4)]:
        if df is not None and 'lr' in df.columns:
            x = x_vals(df)
            y = df['lr'].values
            ax2a.plot(x, y, color=c, linewidth=1.8 if marker is None else 1.2,
                      marker=marker, markersize=ms, alpha=1.0 if marker is None else 0.6,
                      label=lbl if marker is None else None)
            ax2b.plot(x, y, color=c, linewidth=1.8 if marker is None else 1.2,
                      marker=marker, markersize=ms, alpha=1.0 if marker is None else 0.6,
                      label=lbl if marker is None else None)

ax2a.set_title('Learning Rate Schedule  (linear scale)')
ax2a.set_xlabel(x_label); ax2a.set_ylabel('Learning rate')
ax2a.legend(); ax2a.grid(True)
ax2a.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

ax2b.set_title('Learning Rate Schedule  (log scale)')
ax2b.set_xlabel(x_label); ax2b.set_ylabel('Learning rate')
ax2b.set_yscale('log')
ax2b.legend(); ax2b.grid(True)

fig2.suptitle('nanoGPT — Learning Rate', fontsize=14, fontweight='bold')
plt.tight_layout()
lr_path = save.replace('.png', '_lr.png') if save.endswith('.png') else save + '_lr.png'
fig2.savefig(lr_path, dpi=150, bbox_inches='tight')
print(f"Saved: {lr_path}")
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Combined  (loss + LR together, dual y-axis on right panel)
# ─────────────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10))
ax3_step  = axes3[0, 0]   # per-step train loss
ax3_eval  = axes3[0, 1]   # train + val at eval
ax3_lr    = axes3[1, 0]   # LR schedule
ax3_combo = axes3[1, 1]   # val loss + LR on dual axis

for r in runs:
    c, lbl = r['color'], r['label']

    # per-step train loss
    if r['steps'] is not None and 'train_loss' in r['steps'].columns:
        x = x_vals(r['steps'])
        y = r['steps']['train_loss'].values
        ax3_step.plot(x, y, alpha=0.12, color=c, linewidth=0.7)
        ax3_step.plot(x, smooth_series(y, smooth), color=c, linewidth=1.8, label=lbl)

    # eval train + val
    if r['eval'] is not None:
        x = x_vals(r['eval'])
        if 'train_loss' in r['eval'].columns:
            ax3_eval.plot(x, r['eval']['train_loss'].values,
                          color=c, linestyle='--', linewidth=1.4, alpha=0.7,
                          label=f"{lbl} train")
        if 'val_loss' in r['eval'].columns:
            y = r['eval']['val_loss'].values
            ax3_eval.plot(x, y, color=c, linewidth=2, marker='o', markersize=3,
                          label=f"{lbl} val")
            annotate_last(ax3_eval, x, y, c)

    # LR
    for df in [r['steps'], r['eval']]:
        if df is not None and 'lr' in df.columns:
            ax3_lr.plot(x_vals(df), df['lr'].values,
                        color=c, linewidth=1.8, label=lbl)
            break

    # combined val loss + LR dual axis
    if r['eval'] is not None and 'val_loss' in r['eval'].columns:
        x = x_vals(r['eval'])
        ax3_combo.plot(x, r['eval']['val_loss'].values,
                       color=c, linewidth=2, marker='o', markersize=3,
                       label=f"{lbl} val loss")

ax3_combo_lr = ax3_combo.twinx()
for r in runs:
    c, lbl = r['color'], r['label']
    for df in [r['steps'], r['eval']]:
        if df is not None and 'lr' in df.columns:
            ax3_combo_lr.plot(x_vals(df), df['lr'].values,
                              color=c, linewidth=1.2, linestyle=':', alpha=0.6,
                              label=f"{lbl} lr")
            break

ax3_step.set_title(f'Train Loss / step  (smooth={smooth})'); ax3_step.set_xlabel(x_label)
ax3_step.set_ylabel('Loss'); ax3_step.legend(); ax3_step.grid(True)

ax3_eval.set_title('Train vs Val Loss'); ax3_eval.set_xlabel(x_label)
ax3_eval.set_ylabel('Loss'); ax3_eval.legend(); ax3_eval.grid(True)

ax3_lr.set_title('Learning Rate  (log)'); ax3_lr.set_xlabel(x_label)
ax3_lr.set_ylabel('LR'); ax3_lr.set_yscale('log'); ax3_lr.legend(); ax3_lr.grid(True)

ax3_combo.set_title('Val Loss + LR  (dual axis)'); ax3_combo.set_xlabel(x_label)
ax3_combo.set_ylabel('Val loss'); ax3_combo.legend(loc='upper right'); ax3_combo.grid(True)
ax3_combo_lr.set_ylabel('Learning rate', color='grey')
ax3_combo_lr.tick_params(axis='y', labelcolor='grey')

fig3.suptitle('nanoGPT — Combined Training Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()
combined_path = save.replace('.png', '_combined.png') if save.endswith('.png') else save + '_combined.png'
fig3.savefig(combined_path, dpi=150, bbox_inches='tight')
print(f"Saved: {combined_path}")
plt.close(fig3)

print(f"\nDone. 3 files saved:")
print(f"  {loss_path}")
print(f"  {lr_path}")
print(f"  {combined_path}")
