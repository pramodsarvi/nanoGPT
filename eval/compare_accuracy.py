"""
Compare model accuracy between:
  1. Our trained GQA checkpoint  (exp15 — trained from scratch on FineWeb-Edu)
  2. HuggingFace GPT-2 small     (124M, pretrained on WebText ~40B tokens)
  3. HuggingFace GPT-2 medium    (350M, optional — set compare_medium=True)

Metrics:
  - Perplexity on WikiText-103 validation set  (standard LM benchmark)
  - Perplexity on FineWeb-Edu sample           (our training domain)
  - Perplexity on custom prompt list           (qualitative spot check)
  - Side-by-side generation samples

Usage:
  cd /home/pramod/Videos/nanoGPT
  ~/venv/bin/python compare_accuracy.py \
      --ckpt_path=/home/pramod/Downloads/nanoGPT-checkpoints/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu/ckpt.pt

  # Also compare against GPT-2 medium
  ~/venv/bin/python compare_accuracy.py \
      --ckpt_path=/home/pramod/Downloads/nanoGPT-checkpoints/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu/ckpt.pt \
      --compare_medium=True
"""
import os
import sys
import math
import argparse
from contextlib import nullcontext

import torch
import tiktoken
from torch.nn import functional as F

from model_gqa import GPTConfigGQA, GPTGQA

# -----------------------------------------------------------------------------
ckpt_path      = 'out/ckpt.pt'
compare_medium = False       # also load GPT-2 medium (350M)
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype          = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
eval_tokens    = 4096        # tokens to evaluate perplexity over per dataset
stride         = 512         # sliding window stride (overlapping context)
max_gen_tokens = 200         # tokens to generate per sample
temperature    = 0.8
top_k          = 50
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type='cuda', dtype=ptdtype)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True

enc = tiktoken.get_encoding("gpt2")

def separator(title=""):
    w = 68
    if title:
        p = (w - len(title) - 2) // 2
        print(f"\n{'─'*p} {title} {'─'*(w-p-len(title)-2)}")
    else:
        print('─' * w)

# ── load our GQA model ────────────────────────────────────────────────────────
separator("OUR MODEL  (GQA checkpoint)")
print(f"Loading: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu')
model_args = dict(ckpt['model_args'])
model_args['gradient_checkpointing'] = False
model_args['use_triton_attn']        = False

conf     = GPTConfigGQA(**model_args)
our_model = GPTGQA(conf)
sd = {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
      for k, v in ckpt['model'].items()}
our_model.load_state_dict(sd)
our_model.eval().to(device)

n_params = sum(p.numel() for p in our_model.parameters())
print(f"  L{conf.n_layer} H{conf.n_head} KV{conf.n_kv_head} E{conf.n_embd}  "
      f"RoPE={conf.use_rope}  SwiGLU={conf.use_swiglu}")
print(f"  Params: {n_params/1e6:.1f}M  |  iter={ckpt['iter_num']}  "
      f"best_val_loss={ckpt['best_val_loss']:.4f}  "
      f"(ppl={math.exp(ckpt['best_val_loss'].item()):.2f})")

# ── load HF GPT-2 models ──────────────────────────────────────────────────────
try:
    from transformers import GPT2LMHeadModel, GPT2Config
except ImportError:
    print("ERROR: transformers not installed.  pip install transformers")
    sys.exit(1)

def load_hf_gpt2(model_name):
    separator(f"HuggingFace {model_name}")
    print(f"  Loading {model_name} from HuggingFace...")
    m = GPT2LMHeadModel.from_pretrained(model_name)
    m.eval().to(device)
    n = sum(p.numel() for p in m.parameters())
    print(f"  Params: {n/1e6:.1f}M")
    return m

hf_gpt2 = load_hf_gpt2('gpt2')
hf_models = [('HF GPT-2 small (124M, WebText)', hf_gpt2)]

if compare_medium:
    hf_gpt2_medium = load_hf_gpt2('gpt2-medium')
    hf_models.append(('HF GPT-2 medium (350M, WebText)', hf_gpt2_medium))

# ── perplexity helpers ────────────────────────────────────────────────────────

def compute_ppl_our_model(model, token_ids, block_size=1024):
    """Sliding window perplexity for our GQA model."""
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    for start in range(0, len(token_ids) - 1, stride):
        end = min(start + block_size, len(token_ids))
        x = token_ids[start:end - 1].unsqueeze(0).to(device)
        y = token_ids[start + 1:end].unsqueeze(0).to(device)
        if x.shape[1] == 0:
            break
        with torch.no_grad(), ctx:
            _, loss, _ = model(x, y)
        n = x.shape[1]
        total_nll    += loss.item() * n
        total_tokens += n
        if total_tokens >= eval_tokens:
            break
    return math.exp(total_nll / total_tokens), total_tokens

def compute_ppl_hf(model, token_ids, block_size=1024):
    """Sliding window perplexity for HF GPT-2."""
    token_ids = torch.tensor(token_ids, dtype=torch.long)
    total_nll = 0.0
    total_tokens = 0
    model.eval()
    for start in range(0, len(token_ids) - 1, stride):
        end = min(start + block_size, len(token_ids))
        x = token_ids[start:end - 1].unsqueeze(0).to(device)
        y = token_ids[start + 1:end].clone()
        y = y.unsqueeze(0).to(device)
        if x.shape[1] == 0:
            break
        with torch.no_grad(), ctx:
            out  = model(x)
            logits = out.logits   # (1, T, vocab)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
        n = x.shape[1]
        total_nll    += loss.item() * n
        total_tokens += n
        if total_tokens >= eval_tokens:
            break
    return math.exp(total_nll / total_tokens), total_tokens

# ── datasets ──────────────────────────────────────────────────────────────────

def get_wikitext103_tokens():
    """Download WikiText-103 validation set via HuggingFace datasets."""
    try:
        from datasets import load_dataset
        print("  Loading WikiText-103-v1 validation split...")
        ds = load_dataset('wikitext', 'wikitext-103-v1', split='validation', trust_remote_code=True)
        text = '\n'.join(ds['text'])
        tokens = enc.encode(text)
        print(f"  WikiText-103 val: {len(tokens):,} tokens")
        return tokens
    except Exception as e:
        print(f"  Could not load WikiText-103: {e}")
        return None

def get_fineweb_edu_tokens():
    """Stream a small sample from FineWeb-Edu for domain-matched eval."""
    try:
        from datasets import load_dataset
        print("  Loading FineWeb-Edu sample (first 100 docs)...")
        ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                          split='train', streaming=True, trust_remote_code=True)
        texts = []
        for i, ex in enumerate(ds):
            if i >= 100:
                break
            texts.append(ex['text'])
        tokens = enc.encode('\n'.join(texts))
        print(f"  FineWeb-Edu sample: {len(tokens):,} tokens")
        return tokens
    except Exception as e:
        print(f"  Could not load FineWeb-Edu: {e}")
        return None

# ── run perplexity evals ──────────────────────────────────────────────────────

datasets = {}

separator("LOADING EVAL DATASETS")
wt103_tokens = get_wikitext103_tokens()
if wt103_tokens:
    datasets['WikiText-103 val'] = wt103_tokens

fw_tokens = get_fineweb_edu_tokens()
if fw_tokens:
    datasets['FineWeb-Edu sample'] = fw_tokens

# ── perplexity table ──────────────────────────────────────────────────────────

results = {}   # {dataset_name: {model_label: ppl}}

for ds_name, tokens in datasets.items():
    separator(f"PERPLEXITY  —  {ds_name}  ({eval_tokens} tokens, stride={stride})")
    results[ds_name] = {}

    # our model
    ppl, n = compute_ppl_our_model(our_model, tokens)
    label = f"Our GQA exp15 (val_loss={ckpt['best_val_loss']:.3f})"
    results[ds_name][label] = ppl
    print(f"  {label:<52}  PPL = {ppl:.2f}  ({n} tokens)")

    # HF models
    for hf_label, hf_m in hf_models:
        ppl, n = compute_ppl_hf(hf_m, tokens)
        results[ds_name][hf_label] = ppl
        print(f"  {hf_label:<52}  PPL = {ppl:.2f}  ({n} tokens)")

# ── summary table ─────────────────────────────────────────────────────────────
separator("SUMMARY  —  Perplexity (lower is better)")
all_labels = ([f"Our GQA exp15 (val_loss={ckpt['best_val_loss']:.3f})"]
              + [l for l, _ in hf_models])
col_w = 24
header = f"{'Model':<52}" + "".join(f"{ds[:col_w]:>{col_w}}" for ds in results)
print(header)
print('─' * len(header))
for label in all_labels:
    row = f"{label:<52}"
    for ds_name in results:
        ppl = results[ds_name].get(label, float('nan'))
        row += f"{ppl:>{col_w}.2f}"
    print(row)

# ── generation comparison ─────────────────────────────────────────────────────

PROMPTS = [
    "The French Revolution began in",
    "Neural networks learn by",
    "The most important discovery in physics was",
    "To prepare for a job interview, you should",
    "Climate change is caused by",
]

def generate_our(model, prompt, max_new=max_gen_tokens):
    ids   = enc.encode(prompt)
    x     = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), ctx:
        out = model.generate(x, max_new, temperature=temperature, top_k=top_k)
    return enc.decode(out[0].tolist())

def generate_hf(model, prompt, max_new=max_gen_tokens):
    ids   = enc.encode(prompt)
    x     = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), ctx:
        out = model.generate(
            x,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=enc.eot_token,
        )
    return enc.decode(out[0].tolist())

separator("GENERATION SAMPLES")
for prompt in PROMPTS:
    print(f"\nPROMPT: \"{prompt}\"")
    print()

    text = generate_our(our_model, prompt)
    print(f"  [Our GQA exp15]\n  {text}\n")

    for hf_label, hf_m in hf_models:
        text = generate_hf(hf_m, prompt)
        print(f"  [{hf_label}]\n  {text}\n")

    print('·' * 68)

separator()
print("Notes:")
print("  - Our model trained ~15B tokens on FineWeb-Edu (educational web text)")
print("  - HF GPT-2 trained ~40B tokens on WebText (general web text)")
print("  - Lower perplexity on FineWeb-Edu = better fit to our training domain")
print("  - Lower perplexity on WikiText-103 = better general language modeling")
print(f"  - Our model val_loss={ckpt['best_val_loss']:.4f}  "
      f"=> ppl={math.exp(ckpt['best_val_loss'].item()):.2f} on FineWeb-Edu val")
separator()
