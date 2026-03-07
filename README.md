# nanoGPT — GQA + RoPE + SwiGLU

![nanoGPT](assets/nanogpt.jpg)

A fork of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) extended with:

- **Grouped Query Attention (GQA)** — multiple query heads share a single KV head, shrinking the KV cache
- **Rotary Position Embeddings (RoPE)** — replaces learned absolute position embeddings
- **SwiGLU MLP** — LLaMA-style gated activation replacing GELU
- **FSDP training** — multi-GPU training with PyTorch Fully Sharded Data Parallel
- **HuggingFace streaming** — train on FineWeb-Edu without downloading the full dataset
- **Triton GQA kernels** — native prefill/decode kernels without `repeat_interleave` expansion

---

## Repository Layout

```
nanoGPT/
├── model.py                    # Original nanoGPT MHA model
├── model_gqa.py                # GQA + RoPE + SwiGLU model
├── train.py                    # Original single-GPU training script
├── train_fsdp.py               # FSDP multi-GPU training (supports GQA)
├── configurator.py             # Config file loader
├── triton_kernels.py           # Triton kernels for original model
├── triton_kernels_gqa.py       # Native Triton GQA prefill + decode kernels
├── eval/
│   ├── bench_inference.py      # GQA vs MHA KV cache benchmark
│   ├── bench.py                # Original throughput benchmark
│   ├── bench_gqa.py            # GQA throughput benchmark
│   ├── bench_triton_vs_hf.py   # Triton vs HuggingFace attention comparison
│   ├── compare_accuracy.py     # Perplexity vs HuggingFace GPT-2
│   ├── plot_loss.py            # Training curve plots
│   ├── sample.py               # Text generation (original model)
│   └── sample_gqa.py           # Text generation (GQA model)
├── config/
│   └── experiments/            # Per-experiment config files
└── data/
    ├── shakespeare_char/       # Character-level Shakespeare (quick tests)
    └── fineweb_edu/            # Streaming dataset prep for FineWeb-Edu
```

---

## Architecture Changes from nanoGPT

### Grouped Query Attention

Standard MHA gives every query head its own Key and Value head. GQA groups queries so they share KV heads:

```
MHA: n_head=12  →  12 Q heads, 12 K heads, 12 V heads
GQA: n_head=12, n_kv_head=4  →  12 Q heads, 4 K heads, 4 V heads  (3× smaller KV cache)
MQA: n_head=12, n_kv_head=1  →  12 Q heads, 1 K head,  1 V head   (12× smaller KV cache)
```

The fused `c_attn` projection is split into separate `q_proj` and `kv_proj`:

```python
# Original nanoGPT
self.c_attn = nn.Linear(n_embd, 3 * n_embd)

# GQA
self.q_proj  = nn.Linear(n_embd, n_embd)
self.kv_proj = nn.Linear(n_embd, 2 * n_kv_head * head_dim)
```

KV heads are expanded before attention via `repeat_interleave`, or computed natively with the Triton kernels (no extra allocation).

### RoPE

Rotary Position Embeddings rotate Q and K vectors by a position-dependent angle instead of adding a learned embedding to the input:
- No learned `wpe` table — zero extra parameters
- Works correctly with KV caches by rotating at the cache offset position
- Better length generalisation than absolute PE

### SwiGLU MLP

```python
x = F.silu(gate_proj(x)) * up_proj(x)
x = down_proj(x)
```

Hidden dimension is scaled to `(2/3) × 4 × n_embd` to keep parameter count equivalent to the standard MLP.

---

## Experiments

### Exp 01 — GQA (n\_kv=4) + RoPE + SwiGLU on FineWeb-Edu (main run)

**Dataset:** FineWeb-Edu 10BT (streamed from HuggingFace) | **Hardware:** 4× NVIDIA H100 80GB (FSDP)

The primary training run. GPT-2 Small scale (~124M parameters) trained from scratch with all modern architecture improvements.

| Param | Value |
|-------|-------|
| n\_layer / n\_head / n\_kv\_head / n\_embd | 12 / 12 / 4 / 768 |
| block\_size | 1024 |
| use\_rope | True (base=10000) |
| use\_swiglu | True |
| Parameters | ~124M |
| Batch size | 48 seq × 1024 tok = 49,152 tok/step |
| Effective batch | 16 grad accum steps × 49,152 = ~786K tok/step |
| Optimizer | AdamW, lr=1e-3, cosine decay |
| Warmup | 750 iters |
| max\_iters | 40,000 (extended from 20,000 mid-run) |
| Tokens seen at checkpoint | ~14.7B |
| **Best val loss** | **2.9786 (perplexity ≈ 19.7)** |

**Training curve (selected checkpoints):**

| Iter | Train Loss | Val Loss | Tokens Seen |
|------|-----------|---------|-------------|
| 0 | 10.980 | 10.979 | 0 |
| 375 | 4.894 | 4.879 | 276M |
| 750 | 4.034 | 3.970 | 553M |
| 1500 | 3.593 | 3.591 | 1.1B |
| 3000 | 3.334 | 3.382 | 2.2B |
| 6000 | 3.123 | 3.153 | 4.4B |
| 10875 | 3.121 | 3.064 | 8.0B |
| 15750 | 3.046 | 3.008 | 11.6B |
| **19500** | **3.014** | **2.979** | **14.4B** |

![Training loss curve](assets/loss_plot_combined.png)

**Note on mid-run extension:** Training was extended from 20K to 40K iterations at step 19,500 by changing `lr_decay_iters`. The cosine schedule at step 19,500 on a 40K curve has a higher LR than the same step on a 20K curve — effectively a warm restart that can help escape sharp minima.

---

## KV Cache Benchmark: GQA vs MHA

**Script:** `eval/bench_inference.py`
**Hardware:** NVIDIA RTX 4070 Laptop 8GB | **Precision:** bfloat16 | **Batch:** 1

Both variants use the same `model_gqa.py` codebase. MHA is instantiated as `GPTGQA(n_kv_head=n_head)` — identical code path, weights shared for all matching tensors, only KV projection shape differs. This isolates the KV head count effect from any other architectural difference.

**KV cache sizes (full 1024-token context):**

| Variant | KV heads | KV cache / sequence |
|---------|----------|-------------------|
| GQA | 4 | 12.6 MB |
| MHA | 12 | 37.7 MB |
| **Savings** | | **25.1 MB (3.0× smaller)** |

**Results at prompt\_len=512:**

| Metric | GQA (n\_kv=4) | MHA (n\_head=12) |
|--------|--------------|-----------------|
| TTFT (ms) | ~14 ms | ~14 ms |
| Prefill TPS | ~35,000 | ~35,000 |
| Decode TPT (ms/tok) | ~9.1 ms | ~9.5 ms |
| Decode TPS | ~110 | ~105 |
| VRAM prefill (MB) | ~1090 | ~1120 |
| VRAM decode (MB) | ~1100 | ~1135 |

![Decode tokens per second](assets/bench_inference_decode_tps.png)

![Time per decode token](assets/bench_inference_decode_tpt.png)

![VRAM during decode](assets/bench_inference_vram_decode.png)

![GQA vs MHA speedup](assets/bench_inference_speedup_bar.png)

**Key takeaways:**
- **Prefill latency is identical** — compute-bound, not KV-size-bound
- **Decode TPS ~5% higher for GQA** — fewer KV tensors = less memory bandwidth per step
- **VRAM ~30–40 MB lower for GQA** — exactly the KV cache size difference
- At larger batch sizes and longer contexts the gap widens; production systems (LLaMA 2/3, Mistral) report 20–50% decode speedup from GQA

---

## Triton GQA Kernels

`triton_kernels_gqa.py` implements native GQA attention without `repeat_interleave`.

The default `repeat_interleave` approach allocates a full `(B, n_head, T, head_dim)` tensor for K and V before passing to SDPA — storing `n_groups×` more data than necessary. The Triton kernels operate on unexpanded K/V directly:

```
triton_gqa_prefill(q, k, v):
    Q: (B, n_head,    T, head_dim)
    K: (B, n_kv_head, T, head_dim)   # NOT expanded
    V: (B, n_kv_head, T, head_dim)

    Each block handles one (batch, q_head, BLOCK_M query rows)
    kv_head = q_head // n_groups  (computed inside kernel)
    Online softmax (Flash Attention style), BLOCK_M=64, BLOCK_N=32

triton_gqa_decode(q, k, v):
    Q: (B, n_head,    1, head_dim)    # single new token
    K: (B, n_kv_head, S, head_dim)   # full KV cache
    One program per (batch, q_head), attends across all S cached positions
```

---

## Running

```bash
# Reproduce exp15 on 4× H100:
torchrun --standalone --nproc_per_node=4 train_fsdp.py \
    config/experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu.py

# Generate text from checkpoint:
python eval/sample_gqa.py \
    --ckpt_path=/path/to/ckpt.pt \
    --num_samples=5 --max_new_tokens=200

# Run KV cache benchmark:
python eval/bench_inference.py \
    --ckpt_path=/path/to/ckpt.pt

# Plot training curve:
python eval/plot_loss.py \
    --log_csv=/path/to/log.csv
```

---

## References

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
- [Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — HuggingFace
