"""
Sample from a trained GQA or MHA checkpoint.

Usage examples:
  # Basic — generate from a prompt
  python sample_gqa.py --out_dir=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu --start="The history of"

  # Interactive mode — type prompts in a loop
  python sample_gqa.py --out_dir=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu --interactive=True

  # From a file
  python sample_gqa.py --out_dir=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu --start="FILE:prompt.txt"

  # More samples, longer output, lower temperature
  python sample_gqa.py --out_dir=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu \
      --num_samples=5 --max_new_tokens=300 --temperature=0.7 --top_k=50
"""
import os
import sys
from contextlib import nullcontext

import torch
import tiktoken

from model import GPTConfig, GPT
from model_gqa import GPTConfigGQA, GPTGQA

# -----------------------------------------------------------------------------
out_dir      = 'out'           # directory containing ckpt.pt
start        = "\n"            # prompt string, or "FILE:path.txt" to read from file
num_samples  = 3               # number of independent samples to generate
max_new_tokens = 200           # tokens to generate per sample
temperature  = 0.8             # <1 = more focused, >1 = more random
top_k        = 200             # keep only top-k logits (0 = disabled)
seed         = 1337
device       = 'cuda'
dtype        = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile      = False           # torch.compile — faster but slower first run
interactive  = False           # if True, loop for prompts from stdin
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ── load checkpoint ──────────────────────────────────────────────────────────
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    print(f"ERROR: no checkpoint found at {ckpt_path}")
    sys.exit(1)

print(f"Loading checkpoint from {ckpt_path} ...")
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']

# auto-detect GQA vs MHA from saved model_args
is_gqa = 'n_kv_head' in model_args

if is_gqa:
    gptconf = GPTConfigGQA(**model_args)
    model = GPTGQA(gptconf)
    print(f"Model: GQA  n_layer={gptconf.n_layer}  n_head={gptconf.n_head}  "
          f"n_kv_head={gptconf.n_kv_head}  n_embd={gptconf.n_embd}  "
          f"use_rope={gptconf.use_rope}  use_swiglu={gptconf.use_swiglu}")
else:
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    print(f"Model: MHA  n_layer={gptconf.n_layer}  n_head={gptconf.n_head}  n_embd={gptconf.n_embd}")

# strip torch.compile prefix if present
state_dict = checkpoint['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.eval()
model.to(device)

iter_num = checkpoint.get('iter_num', '?')
val_loss = checkpoint.get('best_val_loss', '?')
tokens_seen = checkpoint.get('config', {}).get('max_iters', '?')
print(f"Checkpoint: iter={iter_num}  best_val_loss={val_loss:.4f}" if isinstance(val_loss, float) else f"Checkpoint: iter={iter_num}")

if compile:
    print("Compiling model...")
    model = torch.compile(model)

# ── tokenizer ────────────────────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda ids: enc.decode(ids)

# ── generation helper ─────────────────────────────────────────────────────────
def generate(prompt: str):
    if prompt.startswith('FILE:'):
        with open(prompt[5:], 'r', encoding='utf-8') as f:
            prompt = f.read()

    ids = encode(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        with ctx:
            for i in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature,
                                   top_k=top_k if top_k > 0 else None)
                print(f"\n{'─'*60}  sample {i+1}/{num_samples}")
                print(decode(y[0].tolist()))
    print(f"{'─'*60}")

# ── main ──────────────────────────────────────────────────────────────────────
if interactive:
    print("\nInteractive mode. Type a prompt and press Enter. Ctrl-C or empty line to exit.\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not prompt:
            break
        generate(prompt)
else:
    generate(start)
