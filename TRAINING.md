# Training Guide

## Launch Modes

### 1. Single GPU (default)
```bash
python train_fsdp.py config/experiments/exp11_gqa4_fineweb_scratch.py
```
- No setup needed
- FSDP is disabled automatically (no `RANK` env var)
- Runs as plain PyTorch on one GPU

---

### 2. Multi-GPU, Single Machine (FSDP)
```bash
torchrun --standalone --nproc_per_node=4 train_fsdp.py config/experiments/exp11_gqa4_fineweb_scratch.py
```
- Replace `4` with your GPU count (`nvidia-smi --list-gpus`)
- `--standalone` means all GPUs are on the same machine — no extra networking needed
- Each GPU processes different batches; gradients are synced each step
- `gradient_accumulation_steps` is automatically divided by GPU count, so effective batch size stays the same
- **Requirement:** `gradient_accumulation_steps` must be divisible by GPU count

| GPUs | grad_accum_steps | Works? |
|------|-----------------|--------|
| 2    | 4               | ✅     |
| 4    | 4               | ✅     |
| 3    | 4               | ❌     |
| 3    | 6               | ✅     |

---

### 3. Resume from Checkpoint
No extra flags needed — the experiment configs handle this automatically:

```python
# Inside exp11_gqa4_fineweb_scratch.py
init_from = 'resume' if os.path.exists('out_experiments/exp11.../ckpt.pt') else 'scratch'
```

So re-running the same command resumes automatically:
```bash
python train_fsdp.py config/experiments/exp11_gqa4_fineweb_scratch.py
# or with multiple GPUs:
torchrun --standalone --nproc_per_node=4 train_fsdp.py config/experiments/exp11_gqa4_fineweb_scratch.py
```

---

### 4. Start from Scratch (force)
To ignore an existing checkpoint and retrain from scratch, either:
- Delete the checkpoint: `rm out_experiments/exp11_gqa4_fineweb_scratch/ckpt.pt`
- Or override inline:
```bash
python train_fsdp.py config/experiments/exp11_gqa4_fineweb_scratch.py --init_from=scratch
```

---

### 5. Convert MHA Checkpoint to GQA
Start from a trained standard MHA model and convert its weights to GQA:
```bash
python train_fsdp.py config/my_config.py --init_from=mha_to_gqa --n_kv_head=4
```
K/V heads are created by mean-pooling groups of MHA heads. Q heads are copied directly.

---

### 6. Start from GPT-2 Pretrained Weights (HuggingFace)
```bash
python train_fsdp.py config/my_config.py --init_from=gpt2
# other options: gpt2-medium, gpt2-large, gpt2-xl
```
With GQA enabled, MHA weights are automatically converted:
```bash
python train_fsdp.py config/my_config.py --init_from=gpt2_to_gqa --n_kv_head=4
```

---

## Key Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_kv_head` | `0` | `0` = MHA; positive = GQA (must divide `n_head`) |
| `use_rope` | `True` | RoPE positional encoding (False = learned absolute PE) |
| `use_swiglu` | `False` | SwiGLU MLP (LLaMA-style); False = GELU (GPT-2 style) |
| `lr_scheduler` | `cosine` | `cosine`, `cosine_restarts`, `cyclic_triangular2` |
| `init_from` | `scratch` | `scratch`, `resume`, `mha_to_gqa`, `gpt2`, `gpt2_to_gqa` |

---

## Experiments

| Config | Description |
|--------|-------------|
| `exp11_gqa4_fineweb_scratch.py` | GPT-2 Small, GQA n_kv_head=4, GELU, FineWeb-Edu |
| `exp12_mha_fineweb_scratch.py`  | GPT-2 Small, MHA baseline, GELU, FineWeb-Edu |
| `exp13_gqa4_swiglu_fineweb_scratch.py` | GPT-2 Small, GQA n_kv_head=4, SwiGLU, FineWeb-Edu |

Logs are saved to `out_experiments/<exp_name>/log.csv` with columns:
`iter, train_loss, val_loss, lr, tokens_seen, run_name`
