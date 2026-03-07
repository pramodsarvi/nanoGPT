"""
Side-by-side benchmark: our GQA model vs HuggingFace GPT-2.

Compares:
  - Our GQA (SDPA path)          — default, uses PyTorch scaled_dot_product_attention
  - Our GQA (Triton kernel path) — custom prefill + decode kernels, no KV expand
  - HuggingFace GPT-2            — standard MHA via transformers library

Metrics per model:
  - Parameters (total, non-embedding)
  - Prefill  : ms, tokens/sec
  - Decode   : ms/token, tokens/sec (with KV cache)
  - Peak VRAM: prefill, decode
  - Numerical check: Triton vs SDPA output max-diff

Usage:
  python bench_triton_vs_hf.py --out_dir=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu
  python bench_triton_vs_hf.py --out_dir=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu --prompt_len=256 --decode_steps=100
"""
import os
import sys
import math
import time
from contextlib import nullcontext

import torch

from model_gqa import GPTConfigGQA, GPTGQA

# -----------------------------------------------------------------------------
out_dir      = 'out'
prompt_len   = 128      # tokens in prefill
decode_steps = 50       # autoregressive decode steps
batch_size   = 1
device       = 'cuda'
dtype        = 'bfloat16'
compare_hf   = True     # set False if transformers not installed
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)

torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ── helpers ───────────────────────────────────────────────────────────────────
def fmt(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    return f"{n/1e3:.1f}K"

def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2

def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()

def separator(title=""):
    w = 64
    if title:
        p = (w - len(title) - 2) // 2
        print(f"\n{'─'*p} {title} {'─'*(w-p-len(title)-2)}")
    else:
        print('─' * w)

WARMUP = 5


def bench_model(model, vocab_size, label, use_kv_cache=True):
    """Run prefill + decode benchmark for one model. Returns dict of stats."""
    model.eval()
    prompt = torch.randint(0, vocab_size, (batch_size, prompt_len), device=device)

    # ── prefill ───────────────────────────────────────────────────────────────
    # warmup
    for _ in range(WARMUP):
        with torch.no_grad(), ctx:
            out = model(prompt)
        if isinstance(out, tuple):
            kv = out[2] if len(out) > 2 else None
        else:
            kv = None

    reset_peak()
    t0 = sync_time()
    with torch.no_grad(), ctx:
        out = model(prompt)
    t_prefill = (sync_time() - t0) * 1000
    peak_prefill = peak_mb()

    if isinstance(out, tuple) and len(out) > 2:
        kv_caches = out[2]
    else:
        kv_caches = None
        use_kv_cache = False

    prefill_tps = prompt_len / (t_prefill / 1000)

    # ── decode ────────────────────────────────────────────────────────────────
    curr = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    decode_times = []
    kv = kv_caches

    # warmup decode
    for _ in range(WARMUP):
        with torch.no_grad(), ctx:
            if use_kv_cache and kv is not None:
                out = model(curr, kv_caches=kv)
                kv  = out[2]
            else:
                out = model(torch.randint(0, vocab_size, (batch_size, prompt_len + 1), device=device))
        curr = torch.randint(0, vocab_size, (batch_size, 1), device=device)

    # reset kv cache to post-prefill state
    with torch.no_grad(), ctx:
        out = model(prompt)
    kv = out[2] if isinstance(out, tuple) and len(out) > 2 else None

    reset_peak()
    curr = torch.randint(0, vocab_size, (batch_size, 1), device=device)
    for _ in range(decode_steps):
        t0 = sync_time()
        with torch.no_grad(), ctx:
            if use_kv_cache and kv is not None:
                out = model(curr, kv_caches=kv)
                kv  = out[2]
            else:
                out = model(torch.randint(0, vocab_size, (batch_size, prompt_len + 1), device=device))
        decode_times.append((sync_time() - t0) * 1000)
        curr = torch.randint(0, vocab_size, (batch_size, 1), device=device)

    peak_decode = peak_mb()
    avg_decode_ms = sum(decode_times[WARMUP:]) / len(decode_times[WARMUP:])
    decode_tps = batch_size / (avg_decode_ms / 1000)

    total_params    = sum(p.numel() for p in model.parameters())
    n_embd          = getattr(model.config, 'n_embd', 768)
    vocab            = getattr(model.config, 'vocab_size', vocab_size)
    non_emb_params  = total_params - vocab * n_embd

    return {
        'label':           label,
        'params':          total_params,
        'non_emb_params':  non_emb_params,
        'prefill_ms':      t_prefill,
        'prefill_tps':     prefill_tps,
        'decode_ms':       avg_decode_ms,
        'decode_tps':      decode_tps,
        'peak_prefill_mb': peak_prefill,
        'peak_decode_mb':  peak_decode,
    }


def print_stats(s):
    separator(s['label'])
    print(f"  Params (total)     : {fmt(s['params'])}")
    print(f"  Params (non-emb)   : {fmt(s['non_emb_params'])}")
    print(f"  Prefill {prompt_len} tok   : {s['prefill_ms']:.2f} ms  |  {s['prefill_tps']:.0f} tok/s")
    print(f"  Decode latency     : {s['decode_ms']:.2f} ms/tok  |  {s['decode_tps']:.1f} tok/s")
    print(f"  Peak VRAM prefill  : {s['peak_prefill_mb']:.0f} MB")
    print(f"  Peak VRAM decode   : {s['peak_decode_mb']:.0f} MB")


# ── load GQA checkpoint ───────────────────────────────────────────────────────
separator("LOADING CHECKPOINT")
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    print(f"ERROR: no checkpoint at {ckpt_path}")
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location='cpu')
model_args = dict(ckpt['model_args'])
model_args['gradient_checkpointing'] = False

if 'n_kv_head' not in model_args:
    print("ERROR: checkpoint is MHA, not GQA. This script compares GQA paths.")
    sys.exit(1)

conf = GPTConfigGQA(**model_args)
vocab_size = conf.vocab_size
print(f"  GQA: L{conf.n_layer} H{conf.n_head} KV{conf.n_kv_head} E{conf.n_embd}  "
      f"rope={conf.use_rope}  swiglu={conf.use_swiglu}")

def load_gqa(use_triton_attn):
    cfg = GPTConfigGQA(**{**model_args, 'use_triton_attn': use_triton_attn})
    m = GPTGQA(cfg)
    sd = dict(ckpt['model'])
    for k in list(sd.keys()):
        if k.startswith('_orig_mod.'):
            sd[k[len('_orig_mod.'):]] = sd.pop(k)
    m.load_state_dict(sd)
    return m.to(device).eval()

model_sdpa   = load_gqa(use_triton_attn=False)
model_triton = load_gqa(use_triton_attn=True)

# ── numerical correctness check ───────────────────────────────────────────────
separator("NUMERICAL CHECK  (Triton vs SDPA)")
torch.manual_seed(0)
test_input = torch.randint(0, vocab_size, (1, 64), device=device)
with torch.no_grad(), ctx:
    logits_sdpa,   _, _ = model_sdpa(test_input)
    logits_triton, _, _ = model_triton(test_input)
max_diff = (logits_sdpa - logits_triton).abs().max().item()
mean_diff = (logits_sdpa - logits_triton).abs().mean().item()
print(f"  Logits max  diff: {max_diff:.6f}")
print(f"  Logits mean diff: {mean_diff:.6f}")
print(f"  {'PASS ✓' if max_diff < 0.05 else 'WARN: diff is large — check kernel'}")

# ── run benchmarks ────────────────────────────────────────────────────────────
stats_sdpa   = bench_model(model_sdpa,   vocab_size, "GQA + SDPA (repeat_interleave)")
stats_triton = bench_model(model_triton, vocab_size, "GQA + Triton kernel (no expand)")

all_stats = [stats_sdpa, stats_triton]

# ── HuggingFace GPT-2 ─────────────────────────────────────────────────────────
if compare_hf:
    try:
        from transformers import GPT2LMHeadModel, GPT2Config

        separator("LOADING HF GPT-2")
        # match architecture dims as closely as possible
        hf_cfg = GPT2Config(
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            n_embd=conf.n_embd,
            vocab_size=conf.vocab_size,
            n_positions=conf.block_size,
        )
        hf_model = GPT2LMHeadModel(hf_cfg).to(device).eval()
        n_hf = sum(p.numel() for p in hf_model.parameters())
        print(f"  HF GPT-2 (same dims): {fmt(n_hf)} params")

        # Wrap to match our (logits, loss, kv) return signature
        class HFWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.model = m
                self.config = conf  # use GQA config for param counting

            def forward(self, idx, kv_caches=None):
                past = None
                if kv_caches is not None:
                    # convert our list-of-tuples to HF past_key_values tuple-of-tuples
                    past = tuple(
                        (kv[0].squeeze(0), kv[1].squeeze(0))   # HF wants (n_head, S, head_dim)
                        for kv in kv_caches
                    ) if kv_caches else None
                out = self.model(idx, past_key_values=past, use_cache=True)
                # pack kv into our format: list of (k, v) each (B, n_head, S, head_dim)
                new_kv = [
                    (layer[0].unsqueeze(0), layer[1].unsqueeze(0))
                    for layer in out.past_key_values
                ] if out.past_key_values else None
                return out.logits, None, new_kv

        hf_wrapped = HFWrapper(hf_model)
        stats_hf = bench_model(hf_wrapped, vocab_size, "HuggingFace GPT-2 (MHA, same dims)")
        all_stats.append(stats_hf)
    except ImportError:
        print("  transformers not installed, skipping HF comparison")


# ── summary table ─────────────────────────────────────────────────────────────
for s in all_stats:
    print_stats(s)

separator("SUMMARY TABLE")
header = f"{'Model':<42} {'Params':>8} {'Prefill ms':>12} {'Prefill TPS':>12} {'Decode ms':>10} {'Decode TPS':>11} {'VRAM MB':>9}"
print(header)
print('─' * len(header))
for s in all_stats:
    print(
        f"{s['label']:<42} "
        f"{fmt(s['params']):>8} "
        f"{s['prefill_ms']:>11.2f}ms "
        f"{s['prefill_tps']:>10.0f}  "
        f"{s['decode_ms']:>9.2f}ms "
        f"{s['decode_tps']:>10.1f}  "
        f"{s['peak_decode_mb']:>8.0f}MB"
    )

# ── speedups vs SDPA baseline ─────────────────────────────────────────────────
separator("SPEEDUPS vs GQA+SDPA baseline")
baseline = stats_sdpa
for s in all_stats[1:]:
    pf_x = baseline['prefill_ms'] / s['prefill_ms']
    dc_x = baseline['decode_ms']  / s['decode_ms']
    print(f"  {s['label']}")
    print(f"    Prefill : {pf_x:.2f}x  |  Decode: {dc_x:.2f}x")

separator()
print(f"  prompt_len={prompt_len}  decode_steps={decode_steps}  batch={batch_size}  dtype={dtype}")
separator()
