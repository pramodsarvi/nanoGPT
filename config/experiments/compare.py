"""
Compare training runs by reading all log.csv files under out_experiments/.

Usage:
    ~/venv/bin/python config/experiments/compare.py
    ~/venv/bin/python config/experiments/compare.py --metric val_loss
    ~/venv/bin/python config/experiments/compare.py --plot
"""

import os
import csv
import argparse
from pathlib import Path

def load_logs(root='out_experiments'):
    runs = {}
    for log_path in sorted(Path(root).rglob('log.csv')):
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
    print(f"\n{'Run':<30} {'Final ' + metric:<15} {'Best ' + metric:<15} {'Iters'}")
    print('-' * 70)
    for name, rows in sorted(runs.items()):
        values = [r[metric] for r in rows]
        print(f"{name:<30} {values[-1]:<15.4f} {min(values):<15.4f} {int(rows[-1]['iter'])}")

def plot_runs(runs, metric='val_loss'):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    plt.figure(figsize=(10, 6))
    for name, rows in sorted(runs.items()):
        iters  = [r['iter'] for r in rows]
        values = [r[metric] for r in rows]
        plt.plot(iters, values, marker='o', markersize=3, label=name)

    plt.xlabel('Iteration')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Training Comparison — {metric}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f'out_experiments/comparison_{metric}.png'
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='val_loss', choices=['val_loss', 'train_loss'])
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--root', default='out_experiments')
    args = parser.parse_args()

    runs = load_logs(args.root)
    if not runs:
        print(f"No log.csv files found under {args.root}/")
        print("Run experiments first: bash config/experiments/run_all.sh")
    else:
        print_summary(runs, args.metric)
        if args.plot:
            plot_runs(runs, args.metric)
