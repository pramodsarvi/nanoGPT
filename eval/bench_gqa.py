"""
Benchmark a trained GQA or MHA checkpoint.

Measures:
  - Model parameters (total, non-embedding)
  - Memory: model size, peak training VRAM, peak inference VRAM
  - Training throughput: ms/iter, tokens/sec, MFU %
  - Inference throughput: tokens/sec, ms/token (prefill + decode)
  - KV cache size
  - Perplexity on random batch (sanity check)

Usage:
  python bench_gqa.py --out_dir=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu
  python bench_gqa.py --out_dir=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu --batch_size=8 --block_size=1024
  python bench_gqa.py --out_dir=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu --train_bench=False  # inference only
"""
import os
import sys
import math
import time
from contextlib import nullcontext

import torch
import torch.nn as nn

from model import GPTConfig, GPT
from model_gqa import GPTConfigGQA, GPTGQA

# -----------------------------------------------------------------------------
out_dir       = 'out'
batch_size    = 8
block_size    = 1024       # sequence length for training bench
seed          = 1337
device        = 'cuda'
dtype         = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile       = False      # torch.compile (adds ~1min warmup)
train_bench   = True       # benchmark training step (fwd + bwd + optim)
infer_bench   = True       # benchmark inference (prefill + decode)
train_steps   = 20         # steps to average over (after 10-step warmup)
infer_steps   = 50         # decode steps to average
infer_prompt_len = 128     # tokens in prefill prompt
infer_batch   = 1          # batch size for inference benchmark
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ── helpers ───────────────────────────────────────────────────────────────────
def fmt(n):
    """Human-readable number: 124000000 -> '124.00M'"""
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(n)

def gpu_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2

def gpu_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2

def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()

def separator(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(width - pad - len(title) - 2)}")
    else:
        print(f"{'─'*width}")

# ── load checkpoint ───────────────────────────────────────────────────────────
separator("CHECKPOINT")
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    print(f"ERROR: no checkpoint at {ckpt_path}")
    sys.exit(1)

print(f"Loading: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')
model_args = checkpoint['model_args']
is_gqa = 'n_kv_head' in model_args

if is_gqa:
    # force gradient_checkpointing off for benchmarking
    model_args = dict(model_args)
    model_args['gradient_checkpointing'] = False
    gptconf = GPTConfigGQA(**model_args)
    model = GPTGQA(gptconf)
else:
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

state_dict = checkpoint['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

iter_num = checkpoint.get('iter_num', '?')
best_val_loss = checkpoint.get('best_val_loss', None)
print(f"iter={iter_num}  best_val_loss={best_val_loss:.4f}" if best_val_loss else f"iter={iter_num}")

if compile:
    print("Compiling...")
    model = torch.compile(model)
    # warmup compile
    _x = torch.randint(0, gptconf.vocab_size, (1, 64), device=device)
    with torch.no_grad(), ctx:
        model(_x)
    print("Compiled.")

# ── model stats ───────────────────────────────────────────────────────────────
separator("MODEL PARAMS")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# non-embedding params
emb_params = gptconf.vocab_size * gptconf.n_embd  # wte (shared with lm_head, counted once)
non_emb_params = total_params - emb_params

print(f"  Architecture      : {'GQA' if is_gqa else 'MHA'}")
print(f"  Layers            : {gptconf.n_layer}")
print(f"  Heads (Q)         : {gptconf.n_head}")
if is_gqa:
    print(f"  Heads (KV)        : {gptconf.n_kv_head}  (groups={gptconf.n_head // gptconf.n_kv_head})")
print(f"  Embed dim         : {gptconf.n_embd}")
print(f"  Block size        : {gptconf.block_size}")
print(f"  Vocab size        : {gptconf.vocab_size}")
if is_gqa:
    print(f"  RoPE              : {gptconf.use_rope}  (base={gptconf.rope_base})")
    print(f"  SwiGLU            : {gptconf.use_swiglu}")

print(f"\n  Total params      : {fmt(total_params)}  ({total_params:,})")
print(f"  Non-emb params    : {fmt(non_emb_params)}  ({non_emb_params:,})")
print(f"  Trainable params  : {fmt(trainable_params)}")

# model size on disk / in memory
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
print(f"\n  Model memory      : {(param_bytes + buffer_bytes) / 1024**2:.1f} MB  ({dtype})")
ckpt_size_mb = os.path.getsize(ckpt_path) / 1024**2
print(f"  Checkpoint size   : {ckpt_size_mb:.1f} MB")

# KV cache size per token (for the full block_size context)
if is_gqa:
    head_dim = gptconf.n_embd // gptconf.n_head
    bytes_per_elem = 2  # bf16/fp16
    kv_cache_bytes = (2 * gptconf.n_layer * gptconf.n_kv_head * gptconf.block_size * head_dim * bytes_per_elem)
    kv_cache_mha_bytes = (2 * gptconf.n_layer * gptconf.n_head * gptconf.block_size * head_dim * bytes_per_elem)
    print(f"\n  KV cache (GQA)    : {kv_cache_bytes/1024**2:.2f} MB  per sequence @ block_size={gptconf.block_size}")
    print(f"  KV cache (MHA eq) : {kv_cache_mha_bytes/1024**2:.2f} MB  ({kv_cache_mha_bytes/kv_cache_bytes:.1f}x larger)")
else:
    head_dim = gptconf.n_embd // gptconf.n_head
    bytes_per_elem = 2
    kv_cache_bytes = (2 * gptconf.n_layer * gptconf.n_head * gptconf.block_size * head_dim * bytes_per_elem)
    print(f"\n  KV cache (MHA)    : {kv_cache_bytes/1024**2:.2f} MB  per sequence @ block_size={gptconf.block_size}")

# ── training benchmark ────────────────────────────────────────────────────────
if train_bench:
    separator("TRAINING THROUGHPUT")

    model.train()
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type
    ) if not compile else torch.optim.AdamW(model.parameters(), lr=1e-4)

    # synthetic data
    vocab = gptconf.vocab_size
    X = torch.randint(0, vocab, (batch_size, block_size), device=device)
    Y = torch.randint(0, vocab, (batch_size, block_size), device=device)

    warmup = 10
    times = []
    reset_peak()

    for step in range(warmup + train_steps):
        t0 = sync_time()
        with ctx:
            _, loss, _ = model(X, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        t1 = sync_time()
        if step >= warmup:
            times.append(t1 - t0)

    peak_vram = gpu_peak_mb()
    avg_ms = sum(times) / len(times) * 1000
    tokens_per_sec = (batch_size * block_size) / (avg_ms / 1000)

    # MFU
    if hasattr(model, 'estimate_mfu'):
        mfu = model.estimate_mfu(batch_size, avg_ms / 1000) * 100
    elif hasattr(model, '_orig_mod') and hasattr(model._orig_mod, 'estimate_mfu'):
        mfu = model._orig_mod.estimate_mfu(batch_size, avg_ms / 1000) * 100
    else:
        mfu = float('nan')

    print(f"  batch_size        : {batch_size}  x  block_size={block_size}")
    print(f"  ms / iter         : {avg_ms:.2f} ms")
    print(f"  tokens / sec      : {fmt(int(tokens_per_sec))}")
    print(f"  MFU               : {mfu:.2f}%  (vs H100 bf16 989 TFLOPS)")
    print(f"  Peak VRAM (train) : {peak_vram:.0f} MB  ({peak_vram/1024:.2f} GB)")

    model.eval()

# ── inference benchmark ───────────────────────────────────────────────────────
if infer_bench:
    separator("INFERENCE THROUGHPUT")

    vocab = gptconf.vocab_size
    prompt = torch.randint(0, vocab, (infer_batch, infer_prompt_len), device=device)

    # prefill
    reset_peak()
    torch.cuda.synchronize()
    t0 = sync_time()
    with torch.no_grad(), ctx:
        _, _, kv_caches = model(prompt)
    t_prefill = (sync_time() - t0) * 1000
    peak_vram_prefill = gpu_peak_mb()

    # decode: one token at a time with KV cache
    reset_peak()
    times_decode = []
    curr = prompt[:, [-1]]
    for _ in range(infer_steps):
        t0 = sync_time()
        with torch.no_grad(), ctx:
            _, _, kv_caches = model(curr, kv_caches=kv_caches)
        curr = torch.randint(0, vocab, (infer_batch, 1), device=device)
        times_decode.append((sync_time() - t0) * 1000)

    peak_vram_decode = gpu_peak_mb()
    avg_decode_ms = sum(times_decode) / len(times_decode)
    decode_tps = infer_batch / (avg_decode_ms / 1000)

    print(f"  infer_batch       : {infer_batch}  |  prompt_len={infer_prompt_len}")
    print(f"  Prefill           : {t_prefill:.2f} ms  ({infer_prompt_len} tokens)")
    print(f"  Prefill speed     : {infer_prompt_len / (t_prefill/1000):.0f} tokens/sec")
    print(f"  Decode latency    : {avg_decode_ms:.2f} ms/token")
    print(f"  Decode throughput : {decode_tps:.1f} tokens/sec  (batch={infer_batch})")
    print(f"  Peak VRAM (infer) : {peak_vram_decode:.0f} MB  ({peak_vram_decode/1024:.2f} GB)")

    # inference without KV cache (for comparison)
    reset_peak()
    times_nokv = []
    for _ in range(20):
        seq = torch.randint(0, vocab, (infer_batch, infer_prompt_len + 1), device=device)
        t0 = sync_time()
        with torch.no_grad(), ctx:
            model(seq)
        times_nokv.append((sync_time() - t0) * 1000)
    avg_nokv_ms = sum(times_nokv[5:]) / len(times_nokv[5:])  # skip warmup
    print(f"\n  No-cache fwd      : {avg_nokv_ms:.2f} ms  (len={infer_prompt_len+1})")
    print(f"  KV cache speedup  : {avg_nokv_ms / avg_decode_ms:.1f}x  per decode step")

# ── perplexity sanity check ────────────────────────────────────────────────────
separator("PERPLEXITY (random batch)")
model.eval()
vocab = gptconf.vocab_size
X = torch.randint(0, vocab, (4, min(block_size, gptconf.block_size)), device=device)
Y = torch.randint(0, vocab, (4, min(block_size, gptconf.block_size)), device=device)
with torch.no_grad(), ctx:
    _, loss, _ = model(X, Y)
ppl = math.exp(loss.item())
print(f"  Loss={loss.item():.4f}  Perplexity={ppl:.2f}  (random input — just checks model runs)")

# ── GPU info ──────────────────────────────────────────────────────────────────
separator("GPU")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(device)
    total_vram = props.total_memory / 1024**2
    print(f"  Device            : {props.name}")
    print(f"  Total VRAM        : {total_vram:.0f} MB  ({total_vram/1024:.1f} GB)")
    print(f"  SM count          : {props.multi_processor_count}")
    print(f"  CUDA capability   : {props.major}.{props.minor}")

separator()
