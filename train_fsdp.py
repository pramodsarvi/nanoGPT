"""
This training script is a modified version of train.py that uses PyTorch Fully Sharded Data Parallel (FSDP).
FSDP is a more memory-efficient alternative to DDP that shards model parameters, gradients, and optimizer states across ranks.

To run with FSDP on 4 gpus on 1 node:
$ torchrun --standalone --nproc_per_node=4 train_fsdp.py
"""

import os
import csv
import time
import math
import pickle
import functools
import threading
import queue
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)

from model import GPTConfig, GPT, Block
from model_gqa import GPTConfigGQA, GPTGQA, BlockGQA

# -----------------------------------------------------------------------------
# default config values (same as train.py but FSDP-specific defaults might differ)
out_dir = 'out_fsdp'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
# wandb logging
wandb_log = False
wandb_project = 'nanogpt'
wandb_run_name = ''   # auto-generated from config if empty: e.g. gqa4-rope-L6H6E384
# data
dataset = 'openwebtext'
hf_prefetch_batches = 32    # batches to prefetch in background thread (increase for fast GPUs)
hf_tokenizer_threads = 2   # parallel tokenizer threads (increase for fast GPUs, e.g. 4-8 on H100)
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_kv_head = 0       # 0 = standard MHA (n_kv_head == n_head); set to e.g. 4 for GQA
n_embd = 768
dropout = 0.0
bias = False
rope_base = 10000   # RoPE frequency base; 10000=original, 500000=LLaMA3 long-ctx
use_rope   = True    # False = learned absolute PE (set False when loading GPT-2 pretrained weights)
use_swiglu = False   # True = SwiGLU MLP (LLaMA-style); False = GELU (GPT-2 style)
gradient_checkpointing = False  # recompute block activations on backward; saves ~30-40% memory, ~20% slower
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
lr_scheduler = 'cosine'         # 'cosine' | 'cosine_restarts' | 'cyclic_triangular2'
warmup_iters = 2000
lr_decay_iters = 600000
# cosine_restarts params
lr_restart_period = 2000        # iters per cosine restart cycle (T_0)
lr_restart_mult   = 2           # cycle length multiplier after each restart (T_mult)
lr_restart_decay  = 0.75        # peak LR multiplier per restart (1.0=no decay, 0.5=halve each time)
# cyclic_triangular2 params
lr_cycle_size     = 2000        # iters per half-cycle (step_size_up)
min_lr = 6e-5
# system
backend = 'nccl'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# various inits
fsdp = int(os.environ.get('RANK', -1)) != -1
if fsdp:
    init_process_group(backend=backend)
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master_process = rank == 0
    seed_offset = rank
    assert gradient_accumulation_steps % world_size == 0
    gradient_accumulation_steps //= world_size
else:
    # fallback to single GPU if not run with torchrun
    master_process = True
    seed_offset = 0
    world_size = 1
    local_rank = 0

tokens_per_iter = gradient_accumulation_steps * world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    os.makedirs(out_dir, exist_ok=True)

    # Auto-generate wandb run name from key config dims if not set
    if not wandb_run_name:
        attn_tag = f"gqa{n_kv_head}" if n_kv_head > 0 else "mha"
        rope_tag  = f"rope{rope_base}" if n_kv_head > 0 else "abspe"
        wandb_run_name = f"{attn_tag}-{rope_tag}-L{n_layer}H{n_head}E{n_embd}"
    print(f"run name: {wandb_run_name}")

    # CSV log — written every eval_interval, always on, independent of wandb
    csv_path = os.path.join(out_dir, 'log.csv')
    csv_file = open(csv_path, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:  # write header only for new files
        csv_writer.writerow(['iter', 'train_loss', 'val_loss', 'lr', 'tokens_seen', 'run_name'])
        csv_file.flush()

    # Per-step CSV log — train loss + LR every log_interval steps for visualisation
    steps_csv_path = os.path.join(out_dir, 'log_steps.csv')
    steps_csv_file = open(steps_csv_path, 'a', newline='')
    steps_csv_writer = csv.writer(steps_csv_file)
    if os.path.getsize(steps_csv_path) == 0:
        steps_csv_writer.writerow(['iter', 'train_loss', 'lr', 'tokens_seen'])
        steps_csv_file.flush()

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# FSDP handles its own mixed precision, so we may not need amp.autocast globally 
# but we'll use it for consistency with nanoGPT's forward pass if needed.
# However, the recommended way is MixedPrecision policy.
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loader — two modes:
#   1. local binary (default): reads from data/<dataset>/train.bin via memmap
#   2. HuggingFace streaming: set dataset='hf:<repo>/<name>' e.g. 'hf:HuggingFaceFW/fineweb-edu'
#      streams train AND val directly from HF — zero local disk needed
data_dir = os.path.join('data', dataset.split(':')[-1].split('/')[-1]) if dataset.startswith('hf:') else os.path.join('data', dataset)

# --- HuggingFace streaming iterators with prefetch ---
_HF_VAL_SKIP    = 5000  # skip first N docs for val (same split as prepare.py)

def _make_hf_iter(skip_docs=0):
    import tiktoken
    from datasets import load_dataset as _load_dataset
    hf_path = dataset[3:]  # strip 'hf:'
    if '@' in hf_path:
        hf_path, hf_config = hf_path.split('@', 1)
    else:
        hf_config = 'sample-10BT' if 'fineweb' in hf_path else 'default'
    enc = tiktoken.get_encoding('gpt2')
    eot = enc.eot_token
    ds = _load_dataset(hf_path, name=hf_config, split='train', streaming=True)
    def _token_gen():
        for idx, ex in enumerate(ds):
            if idx < skip_docs:
                continue
            ids = enc.encode_ordinary(ex['text'])
            ids.append(eot)
            yield from ids
    return _token_gen()

class _HFPrefetcher:
    """Fills a queue with pre-tokenized CPU tensors in a single background thread.
    Each rank skips an additional rank*skip_stride docs so ranks see different data.
    docs_consumed is incremented as docs are processed and can be saved/restored via checkpoint.
    """
    def __init__(self, split, resume_docs=0):
        self.split = split
        self.q = queue.Queue(maxsize=hf_prefetch_batches)
        self._stop = threading.Event()
        self.docs_consumed = resume_docs  # tracks total docs processed (saved in checkpoint)
        self._lock = threading.Lock()
        # rank 0: starts at _HF_VAL_SKIP, rank 1: starts at _HF_VAL_SKIP + stride, etc.
        # stride = large prime number of docs to ensure non-overlapping windows
        _rank = int(os.environ.get('RANK', 0))
        _stride = 100003  # ~100K docs between ranks — effectively non-overlapping
        base_skip = 0 if split == 'val' else (_HF_VAL_SKIP + _rank * _stride)
        skip = base_skip + resume_docs  # fast-forward past already-seen docs on resume
        self._thread = threading.Thread(target=self._worker, args=(skip,), daemon=True)
        self._thread.start()

    def _worker(self, skip_docs):
        buf = []
        needed = batch_size * (block_size + 1)
        it = _make_hf_iter(skip_docs=skip_docs)
        docs_this_epoch = 0
        while not self._stop.is_set():
            while len(buf) < needed:
                try:
                    token = next(it)
                    buf.append(token)
                    # count a doc each time we see the EOT token
                    if token == 50256:  # GPT-2 EOT token
                        docs_this_epoch += 1
                        with self._lock:
                            self.docs_consumed += 1
                except StopIteration:
                    it = _make_hf_iter(skip_docs=0)  # wrap around from start
            tokens = torch.tensor(buf[:needed], dtype=torch.long)
            del buf[:needed]
            tokens = tokens.view(batch_size, block_size + 1)
            x = tokens[:, :-1]
            y = tokens[:, 1:]
            self.q.put((x, y))  # blocks if queue is full (natural backpressure)

    def next_batch(self):
        x, y = self.q.get()
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    def get_docs_consumed(self):
        with self._lock:
            return self.docs_consumed

    def stop(self):
        self._stop.set()

_hf_prefetchers = {}  # keyed by split, initialized lazily
_hf_resume_docs = 0  # docs to skip on first init of train prefetcher (set from checkpoint)

def _hf_get_batch(split):
    global _hf_prefetchers
    if split not in _hf_prefetchers:
        resume = _hf_resume_docs if split == 'train' else 0
        _hf_prefetchers[split] = _HFPrefetcher(split, resume_docs=resume)
    return _hf_prefetchers[split].next_batch()

def get_batch(split):
    if dataset.startswith('hf:'):
        return _hf_get_batch(split)
    # local memmap path
    bin_file = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, bin_file), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

iter_num = 0
best_val_loss = 1e9

# vocab size discovery
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']

# model init
# n_kv_head == 0 means standard MHA; any positive value enables GQA
use_gqa = (n_kv_head > 0)
_n_kv_head = n_kv_head if use_gqa else n_head  # resolved value

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if use_gqa:
    model_args['n_kv_head']               = _n_kv_head
    model_args['rope_base']               = rope_base
    model_args['use_rope']                = use_rope
    model_args['use_swiglu']              = use_swiglu
    model_args['gradient_checkpointing']  = gradient_checkpointing

if init_from == 'scratch':
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    if use_gqa:
        gptconf = GPTConfigGQA(**model_args)
        model = GPTGQA(gptconf)
    else:
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

elif init_from in ('resume', 'weights_only'):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model_args = checkpoint['model_args']
    resume_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
    if use_gqa:
        resume_keys.extend(['n_kv_head', 'rope_base', 'use_rope', 'use_swiglu'])
        # gradient_checkpointing is a runtime flag, not saved in checkpoint; apply after load
    for k in resume_keys:
        model_args[k] = checkpoint_model_args[k]
    if use_gqa:
        gptconf = GPTConfigGQA(**model_args)
        model = GPTGQA(gptconf)
    else:
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if init_from == 'weights_only':
        iter_num = 0
        best_val_loss = 1e9
        print("weights_only: loaded model weights, reset iter_num and optimizer state")
    else:
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        _hf_resume_docs = checkpoint.get('hf_docs_consumed', 0)
        if _hf_resume_docs:
            print(f"resuming HF data stream at doc offset {_hf_resume_docs:,}")

elif init_from == 'mha_to_gqa':
    # Load a standard MHA checkpoint and convert to GQA by dropping KV heads
    assert use_gqa, "init_from='mha_to_gqa' requires n_kv_head > 0"
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    model, iter_num, best_val_loss = GPTGQA.from_mha_checkpoint(
        ckpt_path, n_kv_head=_n_kv_head,
        override_args=dict(dropout=dropout, rope_base=rope_base, use_rope=use_rope),
    )

elif init_from.startswith('gpt2'):
    # Load pretrained GPT-2 weights from HuggingFace
    # Supports: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
    # With n_kv_head > 0, converts MHA → GQA after loading
    hf_model_name = init_from.replace('_to_gqa', '')  # 'gpt2_to_gqa' -> 'gpt2'
    print(f"Loading pretrained weights from HuggingFace: {hf_model_name}")
    mha_model = GPT.from_pretrained(hf_model_name, override_args=dict(dropout=dropout))
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(mha_model.config, k)
    if use_gqa:
        # Save MHA checkpoint temporarily, then convert to GQA
        tmp_ckpt = os.path.join(out_dir, 'tmp_mha_pretrained.pt')
        os.makedirs(out_dir, exist_ok=True)
        torch.save({'model': mha_model.state_dict(), 'model_args': vars(mha_model.config)}, tmp_ckpt)
        model, iter_num, best_val_loss = GPTGQA.from_mha_checkpoint(
            tmp_ckpt, n_kv_head=_n_kv_head,
            override_args=dict(dropout=dropout, rope_base=rope_base, use_rope=use_rope),
        )
        os.remove(tmp_ckpt)
    else:
        model = mha_model

# crop block size if needed
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

# apply gradient checkpointing (GQA only; MHA model.py does not support it yet)
if gradient_checkpointing and use_gqa:
    model.config.gradient_checkpointing = True
    if master_process:
        print("gradient checkpointing enabled")

model.to(device)

# FSDP Wrapping
if fsdp:
    # 1. Define Auto Wrap Policy
    block_cls = BlockGQA if use_gqa else Block
    gpt_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={block_cls},
    )
    
    # 2. Define Mixed Precision Policy
    # Note: for float16, we still need ShardedGradScaler
    mixed_precision_policy = MixedPrecision(
        param_dtype=ptdtype,
        reduce_dtype=ptdtype,
        buffer_dtype=ptdtype,
    )
    
    # 3. Wrap Model
    model = FSDP(
        model,
        auto_wrap_policy=gpt_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD, # or SHARD_GRAD_OP for Zero-2
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,   # preserve original param shapes so configure_optimizers works
        limit_all_gathers=True,
    )

# Keep a reference to the FSDP-wrapped model for FSDP-specific operations
# (no_sync, clip_grad_norm_, state_dict_type) since torch.compile() wraps it
fsdp_model = model if fsdp else None

# optimizer
# Note: configure_optimizers should be called on the FSDP model
# so it picks up the sharded (flattened) parameters correctly.
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Load optimizer state if resuming from checkpoint
if init_from == 'resume':
    if fsdp and 'optimizer' in checkpoint:
        # Scatter the full optimizer state dict across FSDP ranks
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
            optim_state = FSDP.optim_state_dict_to_load(
                fsdp_model, optimizer, checkpoint['optimizer']
            )
        optimizer.load_state_dict(optim_state)
        print("loaded FSDP optimizer state from checkpoint")
    elif not fsdp and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("loaded optimizer state from checkpoint")
if init_from in ('resume', 'weights_only'):
    checkpoint = None  # free memory

# GradScaler - use ShardedGradScaler if using FSDP with float16
if dtype == 'float16':
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    scaler = ShardedGradScaler(enabled=True)
else:
    # for bfloat16 or float32, we don't need scaling
    scaler = torch.amp.GradScaler('cuda', enabled=False)

# compile the model AFTER FSDP wrapping
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model, mode='default')

# helps estimate an arbitrarily accurate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    # Average across ranks
    if fsdp:
        for split in out:
            dist.all_reduce(out[split], op=dist.ReduceOp.AVG)
    return out

# learning rate decay scheduler
def get_lr(it):
    # linear warmup shared by all schedulers
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)

    if lr_scheduler == 'cosine':
        # single cosine decay: learning_rate → min_lr over lr_decay_iters
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    elif lr_scheduler == 'cosine_restarts':
        # CosineAnnealingWarmRestarts with decaying peaks (lower highs each restart)
        # T_0=lr_restart_period, T_mult=lr_restart_mult, peak decay=lr_restart_decay
        t = it - warmup_iters
        T_cur = lr_restart_period
        cycle = 0
        while t >= T_cur:
            t -= T_cur
            T_cur = int(T_cur * lr_restart_mult)
            cycle += 1
        peak_lr = learning_rate * (lr_restart_decay ** cycle)  # lower high each cycle
        peak_lr = max(peak_lr, min_lr)
        coeff = 0.5 * (1.0 + math.cos(math.pi * t / T_cur))
        return min_lr + coeff * (peak_lr - min_lr)

    elif lr_scheduler == 'cyclic_triangular2':
        # CyclicLR triangular2: triangular cycles with amplitude halving each cycle
        # step_size = lr_cycle_size (half-cycle length)
        t = it - warmup_iters
        cycle = t // (2 * lr_cycle_size)
        x = abs(t / lr_cycle_size - 2 * cycle - 1)
        scale = 1.0 / (2 ** cycle)   # halve amplitude each full cycle
        return min_lr + (learning_rate - min_lr) * max(0, 1 - x) * scale

    return min_lr

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
# For FSDP, raw_model is the underlying model before wrapping if we want it, 
# but for estimate_mfu we need the config etc.
# For FSDP-specific ops, use fsdp_model (pre-compile reference)
raw_model = model._orig_mod if compile else model

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        # All ranks must participate in estimate_loss() — it contains dist.all_reduce
        losses = estimate_loss()

        # All ranks must participate in FSDP state_dict collection (collective op)
        if fsdp:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state = fsdp_model.state_dict()
                # Gather full optimizer state (all ranks must call; only rank 0 gets data)
                cpu_optim_state = FSDP.optim_state_dict(fsdp_model, optimizer)
        else:
            cpu_state = model.state_dict()
            cpu_optim_state = optimizer.state_dict()

        if master_process:
            tokens_seen = iter_num * tokens_per_iter
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # CSV log — always written
            csv_writer.writerow([
                iter_num,
                f"{losses['train']:.4f}",
                f"{losses['val']:.4f}",
                f"{lr:.6f}",
                tokens_seen,
                wandb_run_name,
            ])
            csv_file.flush()

            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                })

            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = min(best_val_loss, losses['val'])
                # snapshot HF doc offset so we can resume the data stream exactly
                _train_prefetcher = _hf_prefetchers.get('train') if dataset.startswith('hf:') else None
                hf_docs_consumed = _train_prefetcher.get_docs_consumed() if _train_prefetcher else 0
                checkpoint = {
                    'model': cpu_state,
                    'optimizer': cpu_optim_state,
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'hf_docs_consumed': hf_docs_consumed,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # training step
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        # In FSDP, no_sync() is used to prevent gradient synchronization
        is_last_micro_step = (micro_step == gradient_accumulation_steps - 1)
        
        # We only sync on the last micro step
        context = nullcontext() if is_last_micro_step or not fsdp else fsdp_model.no_sync()
        
        with context:
            with ctx:
                logits, loss, _ = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

    # clip gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        if fsdp:
            fsdp_model.clip_grad_norm_(grad_clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        steps_csv_writer.writerow([iter_num, f"{lossf:.4f}", f"{lr:.6f}", iter_num * tokens_per_iter])
        steps_csv_file.flush()
    
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if fsdp:
    destroy_process_group()

if master_process:
    csv_file.close()
    steps_csv_file.close()
