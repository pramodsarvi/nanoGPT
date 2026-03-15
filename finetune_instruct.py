"""
Instruction fine-tuning for a pretrained GPTGQA (or GPT) checkpoint.

Trains with LOSS MASKING: cross-entropy is computed only on response tokens,
not on the instruction/prompt prefix. This teaches the model to follow
instructions rather than memorise the prompt format.

Prompt template (Alpaca-style):
    ### Instruction:
    {instruction}

    ### Input:
    {input}          <- omitted when empty

    ### Response:
    {response}<|endoftext|>

Supported datasets (set via --dataset):
    alpaca           tatsu-lab/alpaca          (52K, HuggingFace)
    tulu2            allenai/tulu-v2-sft-mixture (326K, HuggingFace)
    jsonl:<path>     local JSONL, each line: {"instruction":..., "input":..., "output":...}

Usage:
    # Single GPU — fine-tune from a pretrained checkpoint
    python finetune_instruct.py \\
        --ckpt_path=out_experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu/ckpt.pt \\
        --out_dir=out_instruct/alpaca

    # Multi-GPU (FSDP) — same script, launched with torchrun
    torchrun --standalone --nproc_per_node=4 finetune_instruct.py \\
        --ckpt_path=out_experiments/exp17_1b_gqa_swiglu_rope_multigpu/ckpt.pt \\
        --out_dir=out_instruct/1b_alpaca \\
        --dataset=alpaca \\
        --batch_size=4 \\
        --gradient_accumulation_steps=8

    # Custom JSONL dataset
    python finetune_instruct.py \\
        --ckpt_path=out/ckpt.pt \\
        --dataset=jsonl:data/my_instructions.jsonl
"""

import os
import csv
import time
import math
import json
import random
import functools
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tiktoken

from model import GPTConfig, GPT, Block
from model_gqa import GPTConfigGQA, GPTGQA, BlockGQA

# -----------------------------------------------------------------------------
# Default config (overridable via CLI: --key=value)
# -----------------------------------------------------------------------------
ckpt_path    = ''           # REQUIRED: path to pretrained checkpoint
out_dir      = 'out_instruct'

# dataset
dataset      = 'alpaca'     # 'alpaca' | 'tulu2' | 'jsonl:<path>'
max_seq_len  = 1024         # truncate examples longer than this

# training
batch_size                  = 4
gradient_accumulation_steps = 8
max_iters    = 3000         # total gradient steps (3 epochs over alpaca ≈ 3000 steps)
learning_rate = 2e-5        # low LR for fine-tuning (10–30× smaller than pretrain)
min_lr        = 2e-6
weight_decay  = 0.01
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0
warmup_iters  = 100

# eval / checkpointing
eval_interval          = 200
eval_iters             = 50
log_interval           = 10
always_save_checkpoint = True

# system
device   = 'cuda'
dtype    = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile  = False    # torch.compile off by default for fine-tuning (shorter run)

# wandb
wandb_log      = False
wandb_project  = 'nanogpt-instruct'
wandb_run_name = ''
# -----------------------------------------------------------------------------
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

assert ckpt_path, "ERROR: --ckpt_path is required"

# ── distributed setup ────────────────────────────────────────────────────────
fsdp = int(os.environ.get('RANK', -1)) != -1
if fsdp:
    dist.init_process_group(backend='nccl')
    rank       = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device     = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master_process = (rank == 0)
    seed_offset    = rank
    assert gradient_accumulation_steps % world_size == 0
    gradient_accumulation_steps //= world_size
else:
    master_process = True
    seed_offset    = 0
    world_size     = 1
    local_rank     = 0

torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# ── tokenizer ────────────────────────────────────────────────────────────────
enc     = tiktoken.get_encoding('gpt2')
EOT_ID  = enc.eot_token   # 50256 — used as response end marker
PAD_ID  = -1              # ignored in cross-entropy loss (ignore_index=-1)

# ── prompt formatting ─────────────────────────────────────────────────────────

def format_prompt(instruction: str, inp: str = '') -> str:
    """Format the instruction prefix (everything before the response)."""
    if inp.strip():
        return (
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{inp.strip()}\n\n"
            f"### Response:\n"
        )
    return (
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Response:\n"
    )


def format_example(instruction: str, inp: str, output: str) -> Tuple[List[int], int]:
    """
    Tokenize one example.

    Returns:
        tokens      : full token ids (prompt + response + EOT), length <= max_seq_len
        prompt_len  : number of prompt tokens (loss is MASKED for these positions)
    """
    prompt_str   = format_prompt(instruction, inp)
    response_str = output.strip() + enc.decode([EOT_ID])  # append <|endoftext|>

    prompt_ids   = enc.encode(prompt_str,   allowed_special=set())
    response_ids = enc.encode(response_str, allowed_special={'<|endoftext|>'})

    combined = prompt_ids + response_ids
    if len(combined) > max_seq_len:
        # truncate response (never truncate instruction)
        max_resp = max_seq_len - len(prompt_ids)
        if max_resp <= 0:
            return None, None  # skip: instruction alone exceeds max_seq_len
        response_ids = response_ids[:max_resp]
        combined = prompt_ids + response_ids

    return combined, len(prompt_ids)


# ── dataset loading ──────────────────────────────────────────────────────────

@dataclass
class Example:
    tokens:     List[int]
    prompt_len: int   # loss is masked for tokens[:prompt_len]


def _load_alpaca() -> List[Example]:
    from datasets import load_dataset
    ds = load_dataset('tatsu-lab/alpaca', split='train')
    examples = []
    for row in ds:
        tokens, plen = format_example(row['instruction'], row.get('input', ''), row['output'])
        if tokens is not None and len(tokens) > plen + 1:  # must have at least 1 response token
            examples.append(Example(tokens, plen))
    return examples


def _load_tulu2() -> List[Example]:
    from datasets import load_dataset
    ds = load_dataset('allenai/tulu-v2-sft-mixture', split='train')
    examples = []
    for row in ds:
        # tulu-v2 uses sharegpt-style messages list
        msgs = row.get('messages') or row.get('conversations') or []
        if len(msgs) < 2:
            continue
        instruction = msgs[0].get('content', '') if msgs[0].get('role') == 'user' else ''
        output      = msgs[1].get('content', '') if msgs[1].get('role') == 'assistant' else ''
        if not instruction or not output:
            continue
        tokens, plen = format_example(instruction, '', output)
        if tokens is not None and len(tokens) > plen + 1:
            examples.append(Example(tokens, plen))
    return examples


def _load_jsonl(path: str) -> List[Example]:
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed JSON on line {lineno}: {e}")
                continue
            instruction = row.get('instruction', '')
            inp         = row.get('input', '')
            output      = row.get('output', row.get('response', ''))
            if not instruction or not output:
                continue
            tokens, plen = format_example(instruction, inp, output)
            if tokens is not None and len(tokens) > plen + 1:
                examples.append(Example(tokens, plen))
    return examples


def load_dataset_examples() -> Tuple[List[Example], List[Example]]:
    """Load and split into train/val (95/5)."""
    if master_process:
        print(f"Loading dataset: {dataset} ...")

    if dataset == 'alpaca':
        all_examples = _load_alpaca()
    elif dataset == 'tulu2':
        all_examples = _load_tulu2()
    elif dataset.startswith('jsonl:'):
        all_examples = _load_jsonl(dataset[6:])
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. "
                         "Use 'alpaca', 'tulu2', or 'jsonl:<path>'")

    random.seed(42)
    random.shuffle(all_examples)

    split = max(1, int(len(all_examples) * 0.05))
    val_examples   = all_examples[:split]
    train_examples = all_examples[split:]

    if master_process:
        print(f"  {len(train_examples):,} train  |  {len(val_examples):,} val examples")
        avg_len = sum(len(e.tokens) for e in train_examples) / max(1, len(train_examples))
        print(f"  avg sequence length: {avg_len:.0f} tokens")

    return train_examples, val_examples


# ── batch collation with loss masking ────────────────────────────────────────

def collate_batch(examples: List[Example]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate a list of Examples into (x, y) tensors with loss masking.

    x : (B, T)  input token ids
    y : (B, T)  target token ids; positions in the prompt prefix are set to PAD_ID
                so cross-entropy ignores them (ignore_index=PAD_ID=-1)
    """
    # pad to the length of the longest sequence in the batch
    max_len = max(len(e.tokens) for e in examples)
    max_len = min(max_len, max_seq_len)

    xs, ys = [], []
    for ex in examples:
        toks = ex.tokens[:max_len]
        T    = len(toks)

        # x = toks[:-1], y = toks[1:] with prompt positions masked
        x_toks = toks[:-1]
        y_toks = list(toks[1:])

        # mask prompt positions in y: target is PAD_ID for prompt tokens
        # prompt occupies positions 0..prompt_len-1 in toks
        # those become y positions 0..prompt_len-2 (shifted by 1)
        mask_end = ex.prompt_len - 1   # last masked y index (exclusive)
        for i in range(min(mask_end, len(y_toks))):
            y_toks[i] = PAD_ID

        # pad to max_len-1
        pad_len = (max_len - 1) - len(x_toks)
        x_toks = x_toks + [EOT_ID] * pad_len
        y_toks = y_toks + [PAD_ID] * pad_len

        xs.append(x_toks)
        ys.append(y_toks)

    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


# ── data iterator ─────────────────────────────────────────────────────────────

class DataLoader:
    """Simple infinite iterator that shuffles and batches examples each epoch."""

    def __init__(self, examples: List[Example], batch_sz: int, rank: int = 0, world: int = 1):
        self.examples  = examples
        self.batch_sz  = batch_sz
        self.rank      = rank
        self.world     = world
        self._order    = []
        self._pos      = 0
        self._epoch    = 0
        self._reshuffle()

    def _reshuffle(self):
        self._order = list(range(len(self.examples)))
        # Seed by epoch so all ranks agree on the same shuffle order.
        # Each rank then strides through it to get disjoint batches.
        rng = random.Random(self._epoch)
        rng.shuffle(self._order)
        self._pos = self.rank  # stagger start by rank so ranks see different examples

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = []
        while len(batch) < self.batch_sz:
            if self._pos >= len(self._order):
                self._epoch += 1
                self._reshuffle()
            batch.append(self.examples[self._order[self._pos]])
            self._pos += self.world  # stride by world_size — each rank takes every Nth example
        x, y = collate_batch(batch)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ── load pretrained checkpoint ───────────────────────────────────────────────

if master_process:
    print(f"\nLoading pretrained checkpoint: {ckpt_path}")

checkpoint    = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model_args    = checkpoint['model_args']
is_gqa        = 'n_kv_head' in model_args

if is_gqa:
    gptconf = GPTConfigGQA(**model_args)
    model   = GPTGQA(gptconf)
else:
    gptconf = GPTConfig(**model_args)
    model   = GPT(gptconf)

state_dict = checkpoint['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

if master_process:
    arch = 'GQA' if is_gqa else 'MHA'
    print(f"  {arch}  n_layer={gptconf.n_layer}  n_head={gptconf.n_head}  n_embd={gptconf.n_embd}  "
          + (f"n_kv_head={gptconf.n_kv_head}  " if is_gqa else "")
          + f"params={model.get_num_params()/1e6:.1f}M")
    print(f"  Pretrained at iter={checkpoint.get('iter_num','?')}  "
          f"val_loss={checkpoint.get('best_val_loss', '?')}")

checkpoint = None  # free memory
model.to(device)

# ── FSDP wrapping ────────────────────────────────────────────────────────────

if fsdp:
    block_cls = BlockGQA if is_gqa else Block
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={block_cls},
    )
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=ptdtype, reduce_dtype=ptdtype, buffer_dtype=ptdtype
        ),
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        limit_all_gathers=True,
    )

fsdp_model = model if fsdp else None

# ── optimizer ─────────────────────────────────────────────────────────────────

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# GradScaler — bfloat16 doesn't need scaling; float16 does
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

if compile:
    print("Compiling model...")
    model = torch.compile(model, mode='default')

# ── dataset ───────────────────────────────────────────────────────────────────

# All ranks load the dataset independently (each gets the same shuffle, different batches)
train_examples, val_examples = load_dataset_examples()

train_loader = DataLoader(train_examples, batch_size, rank=local_rank, world=world_size)
val_loader   = DataLoader(val_examples,   batch_size, rank=0,          world=1)

# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ── loss estimation ────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss() -> dict:
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = loader.next_batch()
            with ctx:
                _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    if fsdp:
        for split in out:
            dist.all_reduce(out[split], op=dist.ReduceOp.AVG)
    return out

# ── logging ───────────────────────────────────────────────────────────────────

if master_process:
    if not wandb_run_name:
        ds_tag = dataset.replace('jsonl:', 'jsonl_').replace('/', '_')
        wandb_run_name = f"instruct-{ds_tag}-{gptconf.n_layer}L{gptconf.n_embd}E"

    csv_path  = os.path.join(out_dir, 'log.csv')
    csv_file  = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(['iter', 'train_loss', 'val_loss', 'lr', 'run_name'])
        csv_file.flush()

    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name)

    print(f"\nStarting instruction fine-tuning: {wandb_run_name}")
    print(f"  max_iters={max_iters}  batch={batch_size}  accum={gradient_accumulation_steps}"
          + (f"×{world_size}GPU" if fsdp else "")
          + f"  lr={learning_rate}  dataset={dataset}\n")

# ── training loop ─────────────────────────────────────────────────────────────

iter_num    = 0
best_val_loss = 1e9
t0 = time.time()

while True:
    # set LR
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # ── eval + checkpoint ──────────────────────────────────────────────────
    if iter_num % eval_interval == 0:
        losses = estimate_loss()

        if fsdp:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state      = fsdp_model.state_dict()
                cpu_optim_state = FSDP.optim_state_dict(fsdp_model, optimizer)
        else:
            cpu_state       = model.state_dict()
            cpu_optim_state = optimizer.state_dict()

        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}  val loss {losses['val']:.4f}")
            csv_writer.writerow([iter_num, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                                 f"{lr:.2e}", wandb_run_name])
            csv_file.flush()

            if wandb_log:
                import wandb
                wandb.log({'iter': iter_num, 'train/loss': losses['train'],
                           'val/loss': losses['val'], 'lr': lr})

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = min(best_val_loss, losses['val'])
                ckpt = {
                    'model':      cpu_state,
                    'optimizer':  cpu_optim_state,
                    'model_args': model_args,
                    'iter_num':   iter_num,
                    'best_val_loss': best_val_loss,
                    'instruct_config': {
                        'dataset':      dataset,
                        'max_seq_len':  max_seq_len,
                        'learning_rate': learning_rate,
                        'max_iters':    max_iters,
                    },
                }
                save_path = os.path.join(out_dir, 'ckpt.pt')
                torch.save(ckpt, save_path)
                print(f"  saved checkpoint → {save_path}")

    if iter_num == max_iters:
        break

    # ── gradient step ──────────────────────────────────────────────────────
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(gradient_accumulation_steps):
        is_last = (micro_step == gradient_accumulation_steps - 1)
        context = nullcontext() if is_last or not fsdp else fsdp_model.no_sync()

        with context:
            with ctx:
                X, Y = train_loader.next_batch()
                _, loss, _ = model(X, Y)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        if fsdp:
            fsdp_model.clip_grad_norm_(grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()

    t1   = time.time()
    dt   = t1 - t0
    t0   = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}  lr {lr:.2e}  {dt*1000:.0f}ms")

    iter_num += 1

# ── cleanup ───────────────────────────────────────────────────────────────────

if fsdp:
    dist.destroy_process_group()

if master_process:
    csv_file.close()
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {out_dir}/ckpt.pt")
    print(f"\nGenerate with:")
    print(f"  python eval/sample_gqa.py --out_dir={out_dir} --start=\"### Instruction:\\nExplain what a transformer is.\\n\\n### Response:\\n\"")
