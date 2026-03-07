"""
Inference benchmark: GQA (our model, SDPA + KV cache) vs MHA (HuggingFace GPT-2 + KV cache).

Both variants use KV cache for decode.  The comparison shows the benefit of GQA
(fewer KV heads = smaller KV cache, faster decode) over standard full MHA.

Additionally each GQA variant is run with AND without KV cache to show the raw
impact of caching.

Metrics (per variant, per sequence length):
  TTFT   — Time To First Token  (prefill latency, ms)
  TPS    — Tokens Per Second    (decode throughput)
  TPT    — Time Per Token       (decode latency, ms/tok)
  VRAM   — Peak GPU memory      (MB)
  KV$    — KV cache size        (MB)

Usage:
  cd /home/pramod/Videos/nanoGPT
  ~/venv/bin/python bench_inference.py \
      --ckpt_path=/home/pramod/Downloads/nanoGPT-checkpoints/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu/ckpt.pt

  # Longer sweeps, more decode steps
  ~/venv/bin/python bench_inference.py \
      --ckpt_path=... --decode_steps=200 --batch_size=4
"""
import os, sys, math, time
from contextlib import nullcontext

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from model_gqa import GPTConfigGQA, GPTGQA

# -----------------------------------------------------------------------------
ckpt_path    = 'out/ckpt.pt'
batch_size   = 1
decode_steps = 100          # autoregressive steps per measurement
warmup_steps = 10           # steps discarded before timing
prompt_lens  = [64, 128, 256, 512, 1024]  # sweep over these
device       = 'cuda'
dtype        = 'bfloat16'
save         = 'bench_inference.png'
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

assert torch.cuda.is_available(), "CUDA required"
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
           'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ── helpers ───────────────────────────────────────────────────────────────────
def fmt_n(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.1f}M"
    return f"{n/1e3:.0f}K"

def reset_peak():
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()

def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2

def sync():
    torch.cuda.synchronize(); return time.perf_counter()

def separator(t=""):
    w = 72
    p = (w - len(t) - 2) // 2
    print(f"\n{'─'*p} {t} {'─'*(w-p-len(t)-2)}" if t else '─'*w)

# ── load checkpoint ───────────────────────────────────────────────────────────
separator("CHECKPOINT")
print(f"  {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')
base_args = dict(ckpt['model_args'])
base_args['gradient_checkpointing'] = False
conf = GPTConfigGQA(**base_args)
vocab = conf.vocab_size
n_params = sum(p.numel() for p in GPTGQA(conf).parameters())
print(f"  L{conf.n_layer} H{conf.n_head} KV{conf.n_kv_head} E{conf.n_embd}  "
      f"rope={conf.use_rope}  swiglu={conf.use_swiglu}  params={fmt_n(n_params)}")

def _load_weights(m):
    sd = {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
          for k, v in ckpt['model'].items()}
    m.load_state_dict(sd)
    return m.to(device).eval()

def make_gqa(triton_attn=False):
    args = {**base_args, 'use_triton_attn': triton_attn}
    return _load_weights(GPTGQA(GPTConfigGQA(**args)))

# ── build variants ────────────────────────────────────────────────────────────
separator("BUILDING VARIANTS")

variants = []
# Each variant: label, model, use_kv (bool), color

# 1. GQA + SDPA + KV cache  (our trained model, n_kv_head < n_head)
m_gqa = make_gqa(triton_attn=False)
variants.append({'label': f'GQA  (n_kv={conf.n_kv_head})',
                 'model': m_gqa, 'use_kv': True, 'color': '#2196F3'})
print(f"  [1] GQA SDPA (n_kv_head={conf.n_kv_head})  ready  params={fmt_n(n_params)}")

# 2. MHA — same GPTGQA codebase but n_kv_head=n_head (full MHA, no KV sharing)
# Fair apples-to-apples: identical model code, same weights for all shared
# tensors, only KV head count differs.
def make_mha():
    mha_args = {**base_args, 'n_kv_head': conf.n_head, 'use_triton_attn': False}
    m = GPTGQA(GPTConfigGQA(**mha_args))
    sd_ckpt = {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
               for k, v in ckpt['model'].items()}
    sd_model = m.state_dict()
    sd_model.update({k: v for k, v in sd_ckpt.items()
                     if k in sd_model and sd_model[k].shape == v.shape})
    m.load_state_dict(sd_model)
    return m.to(device).eval()

m_mha = make_mha()
n_mha = sum(p.numel() for p in m_mha.parameters())
variants.append({'label': f'MHA  (n_head={conf.n_head})',
                 'model': m_mha, 'use_kv': True, 'color': '#F44336'})
print(f"  [2] MHA GPTGQA (n_kv_head={conf.n_head}, full MHA)  ready  params={fmt_n(n_mha)}")

# ── kv cache size ─────────────────────────────────────────────────────────────
head_dim     = conf.n_embd // conf.n_head
bytes_per_el = 2  # bf16
kv_gqa_mb = (2 * conf.n_layer * conf.n_kv_head * conf.block_size * head_dim * bytes_per_el) / 1024**2
kv_mha_mb = (2 * conf.n_layer * conf.n_head   * conf.block_size * head_dim * bytes_per_el) / 1024**2

# ── benchmark function ────────────────────────────────────────────────────────

def run_bench(variant, prompt_len):
    model  = variant['model']
    prompt = torch.randint(0, vocab, (batch_size, prompt_len), device=device)

    # ── TTFT: prefill ─────────────────────────────────────────────────────────
    for _ in range(3):   # warmup
        with torch.no_grad(), ctx:
            model(prompt)

    reset_peak()
    t0 = sync()
    with torch.no_grad(), ctx:
        _, _, kv = model(prompt)
    ttft_ms      = (sync() - t0) * 1000
    vram_prefill = peak_mb()

    # ── Decode: TPS / TPT  (KV cache) ─────────────────────────────────────────
    # warmup
    kv_w = kv
    for _ in range(warmup_steps):
        curr = torch.randint(0, vocab, (batch_size, 1), device=device)
        with torch.no_grad(), ctx:
            _, _, kv_w = model(curr, kv_caches=kv_w)

    # fresh kv for timed decode
    with torch.no_grad(), ctx:
        _, _, kv = model(prompt)

    reset_peak()
    times = []
    for _ in range(decode_steps):
        curr = torch.randint(0, vocab, (batch_size, 1), device=device)
        t0 = sync()
        with torch.no_grad(), ctx:
            _, _, kv = model(curr, kv_caches=kv)
        times.append((sync() - t0) * 1000)

    vram_decode = peak_mb()
    tpt_ms  = float(np.mean(times[warmup_steps:]))   # ms per token
    tps     = batch_size / (tpt_ms / 1000)
    prefill_tps = (prompt_len * batch_size) / (ttft_ms / 1000)

    return {
        'ttft_ms':       ttft_ms,
        'prefill_tps':   prefill_tps,
        'tpt_ms':        tpt_ms,
        'tps':           tps,
        'vram_prefill':  vram_prefill,
        'vram_decode':   vram_decode,
    }

# ── run sweep ──────────────────────────────────────────────────────────────────
separator("BENCHMARKING  (KV cache enabled)")
results  = {v['label']: {} for v in variants}
valid_lens = [p for p in prompt_lens if p <= conf.block_size]

for v in variants:
    print(f"\n  {v['label']}")
    for plen in valid_lens:
        r = run_bench(v, plen)
        results[v['label']][plen] = r
        print(f"    prompt={plen:4d}  TTFT={r['ttft_ms']:6.1f}ms  "
              f"prefill={r['prefill_tps']:6.0f}tok/s  "
              f"TPT={r['tpt_ms']:5.2f}ms  TPS={r['tps']:5.1f}  "
              f"VRAM={r['vram_decode']:.0f}MB")

# ── results table ─────────────────────────────────────────────────────────────
separator("RESULTS TABLE")
col = 28
metric_defs = [
    ('TTFT (ms)',       'ttft_ms',      '{:.1f}'),
    ('Prefill TPS',     'prefill_tps',  '{:.0f}'),
    ('Decode TPT (ms)', 'tpt_ms',       '{:.2f}'),
    ('Decode TPS',      'tps',          '{:.1f}'),
    ('VRAM prefill MB', 'vram_prefill', '{:.0f}'),
    ('VRAM decode MB',  'vram_decode',  '{:.0f}'),
]
for plen in valid_lens:
    print(f"\n  prompt_len={plen}  batch={batch_size}  decode_steps={decode_steps}")
    hdr = f"  {'Metric':<22}" + "".join(f"{v['label']:>{col}}" for v in variants)
    print(hdr)
    print('  ' + '─' * (22 + col * len(variants)))
    for m_label, m_key, m_fmt in metric_defs:
        row = f"  {m_label:<22}"
        for v in variants:
            val = results[v['label']].get(plen, {}).get(m_key, float('nan'))
            row += f"{m_fmt.format(val):>{col}}"
        print(row)

# ── KV cache sizes ─────────────────────────────────────────────────────────────
separator("KV CACHE SIZES")
print(f"  GQA  n_kv_head={conf.n_kv_head}  →  {kv_gqa_mb:.2f} MB  (block_size={conf.block_size})")
print(f"  MHA  n_head   ={conf.n_head}  →  {kv_mha_mb:.2f} MB")
print(f"  GQA saves {kv_mha_mb - kv_gqa_mb:.2f} MB  ({kv_mha_mb/kv_gqa_mb:.1f}x smaller)")

# ── speedup table ─────────────────────────────────────────────────────────────
separator("GQA  vs  MHA  speedup")
gqa_label = next(v['label'] for v in variants if v['label'].startswith('GQA'))
mha_label = next((v['label'] for v in variants if v['label'].startswith('MHA')), None)
if mha_label:
    for plen in valid_lens:
        g = results[gqa_label].get(plen)
        m = results[mha_label].get(plen)
        if not g or not m:
            continue
        print(f"  prompt={plen:4d}  decode TPS {g['tps']/m['tps']:.2f}x  "
              f"TTFT {m['ttft_ms']/g['ttft_ms']:.2f}x  "
              f"VRAM {m['vram_decode']-g['vram_decode']:+.0f}MB")

gpu = torch.cuda.get_device_properties(device)

# ── shared plot style ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                            'axes.titlesize': 13, 'axes.labelsize': 11,
                            'legend.fontsize': 10, 'grid.alpha': 0.3})
save_base = save.replace('.png', '')

def _line(ax, key, ylabel, title):
    for v in variants:
        xs = [p for p in valid_lens if results[v['label']].get(p)]
        ys = [results[v['label']][p][key] for p in xs]
        ax.plot(xs, ys, marker='o', markersize=7, linewidth=2.4,
                color=v['color'], label=v['label'])
        if ys:
            ax.annotate(f"{ys[-1]:.1f}", xy=(xs[-1], ys[-1]),
                        xytext=(6, 3), textcoords='offset points',
                        fontsize=9, color=v['color'])
    ax.set_xlabel('Prompt length (tokens)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    ax.set_xticks(valid_lens)

subtitle = (f"GQA n_kv={conf.n_kv_head} vs MHA n_head={conf.n_head}  |  "
            f"GPU: {gpu.name}  |  batch={batch_size}  |  {dtype}  |  KV cache")

separator("SAVING PLOTS")

# Plot 1 — TTFT
fig, ax = plt.subplots(figsize=(8, 5))
_line(ax, 'ttft_ms', 'ms', 'Time To First Token (TTFT)')
fig.suptitle(subtitle, fontsize=10)
plt.tight_layout()
p = f"{save_base}_ttft.png"
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}")

# Plot 2 — Decode TPS
fig, ax = plt.subplots(figsize=(8, 5))
_line(ax, 'tps', 'tokens / sec', 'Decode Throughput (TPS)')
fig.suptitle(subtitle, fontsize=10)
plt.tight_layout()
p = f"{save_base}_decode_tps.png"
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}")

# Plot 3 — Decode TPT
fig, ax = plt.subplots(figsize=(8, 5))
_line(ax, 'tpt_ms', 'ms / token', 'Decode Latency (TPT)')
fig.suptitle(subtitle, fontsize=10)
plt.tight_layout()
p = f"{save_base}_decode_tpt.png"
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}")

# Plot 4 — Prefill TPS
fig, ax = plt.subplots(figsize=(8, 5))
_line(ax, 'prefill_tps', 'tokens / sec', 'Prefill Throughput (TPS)')
fig.suptitle(subtitle, fontsize=10)
plt.tight_layout()
p = f"{save_base}_prefill_tps.png"
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}")

# Plot 5 — VRAM prefill
fig, ax = plt.subplots(figsize=(8, 5))
_line(ax, 'vram_prefill', 'MB', 'Peak VRAM — Prefill')
fig.suptitle(subtitle, fontsize=10)
plt.tight_layout()
p = f"{save_base}_vram_prefill.png"
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}")

# Plot 6 — VRAM decode
fig, ax = plt.subplots(figsize=(8, 5))
_line(ax, 'vram_decode', 'MB', 'Peak VRAM — Decode')
fig.suptitle(subtitle, fontsize=10)
plt.tight_layout()
p = f"{save_base}_vram_decode.png"
fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
print(f"  Saved: {p}")

# Plot 7 — GQA vs MHA decode TPS bar chart with speedup annotations
if mha_label:
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(valid_lens))
    bar_w = 0.35
    gqa_tps = [results[gqa_label][p]['tps'] for p in valid_lens]
    mha_tps = [results[mha_label][p]['tps'] for p in valid_lens]
    ax.bar(x_pos - bar_w/2, gqa_tps, bar_w, label=gqa_label, color='#2196F3', alpha=0.88)
    ax.bar(x_pos + bar_w/2, mha_tps, bar_w, label=mha_label, color='#F44336', alpha=0.88)
    for i, (g, m) in enumerate(zip(gqa_tps, mha_tps)):
        if m > 0:
            ax.text(x_pos[i], max(g, m) * 1.02, f"{g/m:.2f}×",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#222')
    ax.set_title('GQA vs MHA — Decode TPS  (ratio = GQA / MHA)')
    ax.set_xlabel('Prompt length (tokens)')
    ax.set_ylabel('Decode TPS')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(p) for p in valid_lens])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.suptitle(subtitle, fontsize=10)
    plt.tight_layout()
    p = f"{save_base}_speedup_bar.png"
    fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Saved: {p}")

separator()
print(f"  GPU  : {gpu.name}  ({gpu.total_memory/1024**3:.0f} GB)")
print(f"  dtype: {dtype}  |  batch: {batch_size}  |  decode steps: {decode_steps}")
print(f"  GQA KV cache: {kv_gqa_mb:.1f} MB  |  MHA KV cache: {kv_mha_mb:.1f} MB  "
      f"({kv_mha_mb/kv_gqa_mb:.1f}x)")
separator()
