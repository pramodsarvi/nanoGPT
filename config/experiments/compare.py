"""
Compare training runs by reading all log.csv files under out_experiments/.

Usage:
    ~/venv/bin/python config/experiments/compare.py
    ~/venv/bin/python config/experiments/compare.py --metric val_loss
    ~/venv/bin/python config/experiments/compare.py --plot
    ~/venv/bin/python config/experiments/compare.py --plot --steps   # per-step loss + LR
"""

import os
import csv
import argparse
from pathlib import Path

def load_logs(root='out_experiments', filename='log.csv'):
    runs = {}
    for log_path in sorted(Path(root).rglob(filename)):
        rows = []
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) if k not in ('run_name',) else v
                             for k, v in row.items()})
        if rows:
            run_name = rows[0].get('run_name', log_path.parent.name)
            runs[run_name] = rows
    return runs

def print_summary(runs, metric='val_loss'):
    print(f"\n{'Run':<40} {'Final ' + metric:<15} {'Best ' + metric:<15} {'Iters'}")
    print('-' * 80)
    for name, rows in sorted(runs.items()):
        values = [r[metric] for r in rows if metric in r]
        if not values:
            continue
        print(f"{name:<40} {values[-1]:<15.4f} {min(values):<15.4f} {int(rows[-1]['iter'])}")

def plot_runs(runs, metric='val_loss'):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    plt.figure(figsize=(10, 6))
    for name, rows in sorted(runs.items()):
        iters  = [r['iter'] for r in rows if metric in r]
        values = [r[metric] for r in rows if metric in r]
        plt.plot(iters, values, marker='o', markersize=3, label=name)

    plt.xlabel('Iteration')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Training Comparison — {metric}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f'out_experiments/comparison_{metric}.png'
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.show()

def plot_steps(runs):
    """Plot per-step train loss and LR from log_steps.csv."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, rows in sorted(runs.items()):
        iters  = [r['iter'] for r in rows]
        losses = [r['train_loss'] for r in rows]
        lrs    = [r['lr'] for r in rows]
        ax1.plot(iters, losses, linewidth=0.8, label=name)
        ax2.plot(iters, lrs,    linewidth=0.8, label=name)

    ax1.set_ylabel('Train Loss')
    ax1.set_title('Per-step Train Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel('Learning Rate')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = 'out_experiments/comparison_steps.png'
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='val_loss', choices=['val_loss', 'train_loss'])
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--steps', action='store_true', help='Plot per-step loss+LR from log_steps.csv')
    parser.add_argument('--root', default='out_experiments')
    args = parser.parse_args()

    if args.steps:
        runs = load_logs(args.root, filename='log_steps.csv')
        if not runs:
            print(f"No log_steps.csv files found under {args.root}/")
        else:
            plot_steps(runs)
    else:
        runs = load_logs(args.root, filename='log.csv')
        if not runs:
            print(f"No log.csv files found under {args.root}/")
            print("Run experiments first: bash config/experiments/run_all.sh")
        else:
            print_summary(runs, args.metric)
            if args.plot:
                plot_runs(runs, args.metric)
