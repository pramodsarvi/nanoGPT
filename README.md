
# nanoGPT — Extended

![nanoGPT](assets/nanogpt.jpg)

This is an extended fork of [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) adding:

- **Grouped Query Attention (GQA)** with configurable KV heads
- **Rotary Position Embeddings (RoPE)** replacing learned absolute positional embeddings
- **FSDP training** (Fully Sharded Data Parallel) for multi-GPU runs
- **FineWeb-Edu streaming data prep** — no full dataset download required
- **Experiment infrastructure** — named configs, CSV logging, comparison script

---

## Environment

```bash
# Python 3.12, venv at ~/venv
~/venv/bin/pip install torch numpy transformers datasets tiktoken wandb tqdm
```

---

## File Overview

| File | Description |
|---|---|
| `model.py` | Original GPT: MHA + learned absolute position embeddings + KV cache + Triton kernels |
| `model_gqa.py` | **New**: GPT with GQA + RoPE (see details below) |
| `train.py` | Original single/multi-GPU training script (DDP) |
| `train_fsdp.py` | **New/Modified**: FSDP training script; supports both MHA and GQA models |
| `sample.py` | Sampling / inference script |
| `bench.py` | Quick benchmark of the training loop |
| `configurator.py` | Config override mechanism used by all train scripts |
| `triton_kernels.py` | Optional Triton kernels for LayerNorm, GELU, attention |

**Data:**

| Path | Description |
|---|---|
| `data/shakespeare_char/` | Character-level Shakespeare — tiny, for quick tests |
| `data/shakespeare/` | BPE-tokenized Shakespeare — for finetuning from GPT-2 |
| `data/openwebtext/` | OpenWebText — classic nanoGPT benchmark (~9B tokens, ~70GB) |
| `data/fineweb_edu/` | **New**: FineWeb-Edu 10BT — higher quality, streamed from HuggingFace |

**Configs:**

| Path | Description |
|---|---|
| `config/train_shakespeare_char.py` | Original Shakespeare char config (MHA) |
| `config/train_shakespeare_gqa.py` | Shakespeare char with GQA + RoPE |
| `config/train_gpt2.py` | GPT-2 124M reproduction config |
| `config/experiments/` | **New**: Named experiment configs for architecture comparison |

---

## What's New

### 1. Grouped Query Attention (`model_gqa.py`)

GQA reduces the number of Key/Value heads while keeping full Query heads. Multiple query heads share a single KV head, reducing KV cache memory at inference time.

**Key config param:** `n_kv_head`

| `n_kv_head` | Mode | Description |
|---|---|---|
| `n_head` | MHA | Standard multi-head attention (no sharing) |
| `1` | MQA | All query heads share one KV head — maximum compression |
| `2–n_head-1` | GQA | Intermediate — e.g. `n_kv_head=4` with `n_head=12` → 3 queries per KV head |

**Projection sizes:**
- `q_proj`: `n_embd → n_embd` (all query heads)
- `kv_proj`: `n_embd → 2 × n_kv_head × head_dim` (K + V, reduced)
- `c_proj`: `n_embd → n_embd` (output, unchanged)

KV heads are expanded to match query heads via `repeat_interleave` before attention — no extra parameters, just a view.

**KV cache** shape is `(B, n_kv_head, T, head_dim)` — smaller than standard MHA.

### 2. Rotary Position Embeddings (RoPE)

RoPE replaces the learned `wpe` embedding table. Position information is encoded directly into Q and K via rotation, not added to token embeddings.

**Benefits over absolute PE:**
- No learned parameters for position (saves `block_size × n_embd` params)
- Better length generalisation
- Position offset-aware during KV cache decoding — each new token gets the correct absolute position automatically

**Key config param:** `rope_base` (default `10000`, set `500000` for LLaMA 3-style long context)

RoPE is applied **only to Q and K**, never to V — matching the original paper and all modern implementations.

### 3. Loading MHA → GQA (`from_mha_checkpoint`)

```python
model, iter_num, best_val_loss = GPTGQA.from_mha_checkpoint(
    ckpt_path='out/ckpt.pt',
    n_kv_head=4,
)
```

Weight mapping from a standard nanoGPT MHA checkpoint:

| MHA weight | Shape | GQA destination | How |
|---|---|---|---|
| `attn.c_attn.weight` | `[3C, C]` | `q_proj.weight` | First `C` rows (Q block) |
| `attn.c_attn.weight` | `[3C, C]` | `kv_proj.weight` | Rows `C:C+kv_dim` and `2C:2C+kv_dim` (first `n_kv_head` heads of K and V) |
| `attn.c_proj.weight` | `[C, C]` | `c_proj.weight` | Direct copy |
| `transformer.wpe` | — | *(discarded)* | RoPE has no learned weights |
| All others (ln, mlp, wte, ln_f) | — | Same key | Direct copy |

### 4. FSDP Training (`train_fsdp.py`)

Supports both MHA and GQA models, single GPU and multi-GPU.

**Run on multiple GPUs:**
```bash
torchrun --standalone --nproc_per_node=4 train_fsdp.py config/train_gpt2.py --n_kv_head=4
```

**Run on single GPU (FSDP disabled, useful for testing):**
```bash
~/venv/bin/python train_fsdp.py config/train_shakespeare_gqa.py
```

**All configurable parameters:**

| Parameter | Default | Description |
|---|---|---|
| `n_kv_head` | `0` | `0` = MHA; positive value = GQA with that many KV heads |
| `rope_base` | `10000` | RoPE frequency base |
| `init_from` | `scratch` | `scratch`, `resume`, or `mha_to_gqa` |
| `dataset` | `openwebtext` | Subdirectory under `data/` |
| `wandb_log` | `False` | Enable W&B logging |
| `wandb_run_name` | `''` | Auto-generated if empty: e.g. `gqa4-rope10000-L12H12E768` |

**`init_from` modes:**

| Value | Behaviour |
|---|---|
| `scratch` | Train from random init |
| `resume` | Resume from `out_dir/ckpt.pt` |
| `mha_to_gqa` | Load MHA checkpoint, slice KV heads to `n_kv_head`, continue training |

### 5. FineWeb-Edu Streaming Data Prep

Streams the `sample-10BT` split of FineWeb-Edu directly from HuggingFace — no need to download the full dataset (~25GB saved).

```bash
~/venv/bin/python data/fineweb_edu/prepare.py
```

Outputs:
- `data/fineweb_edu/train.bin` — ~17–19GB, ~10B GPT-2 BPE tokens
- `data/fineweb_edu/val.bin` — ~8MB (first 5000 documents)
- `data/fineweb_edu/meta.pkl` — `vocab_size=50257` for auto-discovery

Then train with:
```bash
~/venv/bin/python train_fsdp.py --dataset=fineweb_edu --n_kv_head=4
```

### 6. Experiment Infrastructure

Run multiple named configs and compare their training curves.

**Available experiment configs (`config/experiments/`):**

| Config | Architecture | `n_kv_head` | `rope_base` |
|---|---|---|---|
| `exp01_mha_abspe.py` | MHA + absolute PE (baseline) | — | — |
| `exp02_gqa2_rope.py` | GQA + RoPE | 2 | 10000 |
| `exp03_gqa3_rope.py` | GQA + RoPE | 3 | 10000 |
| `exp04_mqa_rope.py` | MQA + RoPE | 1 | 10000 |
| `exp05_gqa2_rope500k.py` | GQA + RoPE (long-ctx base) | 2 | 500000 |

All experiments use the same model size (L6 H6 E384) and dataset (shakespeare_char) for fair comparison.

**Run all sequentially:**
```bash
bash config/experiments/run_all.sh
```

**Compare results:**
```bash
# Print summary table
~/venv/bin/python config/experiments/compare.py

# With plot (requires matplotlib)
~/venv/bin/python config/experiments/compare.py --plot
```

### 7. CSV Logging

Every training run writes a `log.csv` to `out_dir/` automatically — no wandb required.

```
iter,train_loss,val_loss,lr,tokens_seen,run_name
0,4.2314,4.2184,0.000010,0,exp02-gqa2-rope10k
250,1.7828,1.9031,0.000986,1024000,exp02-gqa2-rope10k
...
```

Load with pandas for custom analysis:
```python
import pandas as pd
df = pd.read_csv('out_experiments/exp02_gqa2_rope/log.csv')
```

---

## Quick Start

**1. Validate GQA implementation (Shakespeare char, ~5 min):**
```bash
~/venv/bin/python data/shakespeare_char/prepare.py
~/venv/bin/python train_fsdp.py config/train_shakespeare_gqa.py
```

**2. Run all architecture comparison experiments (~15 min total):**
```bash
bash config/experiments/run_all.sh
~/venv/bin/python config/experiments/compare.py --plot
```

**3. Train GQA on FineWeb-Edu (stream data, then train):**
```bash
# Step 1: stream and tokenize (~1–3 hrs, network-bound)
~/venv/bin/python data/fineweb_edu/prepare.py

# Step 2: train GQA model
~/venv/bin/python train_fsdp.py \
  --dataset=fineweb_edu \
  --n_kv_head=4 \
  --n_layer=12 --n_head=12 --n_embd=768 \
  --out_dir=out_fineweb_gqa
```

**4. Convert an existing MHA checkpoint to GQA:**
```bash
~/venv/bin/python train_fsdp.py \
  --init_from=mha_to_gqa \
  --n_kv_head=4 \
  --out_dir=out_mha_run \
  --out_dir=out_gqa_finetuned
```

---

## Original nanoGPT

The original nanoGPT code (`model.py`, `train.py`, `sample.py`) is preserved unchanged. All additions are additive — `model_gqa.py` and `train_fsdp.py` are separate files and do not modify the original training path.

For the original README and usage, see the [upstream repo](https://github.com/karpathy/nanoGPT).
