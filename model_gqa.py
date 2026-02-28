"""
GPT with Grouped Query Attention (GQA) + Rotary Position Embeddings (RoPE).

Key differences from model.py:
- n_kv_head < n_head: multiple query heads share one K/V head
- Separate q_proj and kv_proj instead of fused c_attn
- KV cache shape is (B, n_kv_head, T, hs) — smaller than MHA
- RoPE replaces learned absolute position embeddings (wpe removed)
  - Applied to Q and K only, inside attention, before SDPA
  - Position-offset-aware for KV cache decoding
- from_mha_checkpoint(): loads a standard MHA checkpoint, drops KV heads
  by keeping the first n_kv_head heads (slicing strategy).

References:
- GQA paper: https://arxiv.org/abs/2305.13245
- RoPE paper: https://arxiv.org/abs/2104.09864
- LLaMA 2/3 use both GQA and RoPE
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from triton_kernels import triton_layernorm, triton_layernorm_residual, TritonGELU, triton_gelu_bias
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# -----------------------------------------------------------------------------
# RoPE
# -----------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Precomputes RoPE cos/sin tables up to max_seq_len.
    Registered as buffers (not parameters — no learned weights).

    apply_rotary() is a standalone function used inside attention
    so it can be applied separately to Q and K with different offsets.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        # θ_i = 1 / (base^(2i/d))  for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cos/sin for all positions up to max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, head_dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)      # (seq_len, head_dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
        self._max_cached = seq_len

    def get_cos_sin(self, seq_len: int, offset: int = 0, device=None):
        """Return cos/sin for positions [offset, offset+seq_len)."""
        needed = offset + seq_len
        if needed > self._max_cached:
            self._build_cache(needed)
        cos = self.cos_cached[offset : offset + seq_len]  # (seq_len, head_dim)
        sin = self.sin_cached[offset : offset + seq_len]
        return cos, sin


def rotate_half(x):
    """Rotate the second half of the last dimension: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x, cos, sin):
    """
    Apply RoPE to tensor x of shape (B, n_head, T, head_dim).
    cos/sin shape: (T, head_dim) — broadcast over B and n_head.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# -----------------------------------------------------------------------------
# Layers
# -----------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias, use_triton=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.use_triton = HAS_TRITON and use_triton
        self.eps = 1e-5

    def forward(self, input):
        if self.use_triton and input.is_cuda:
            bias = self.bias if self.bias is not None else torch.zeros_like(self.weight)
            return triton_layernorm(input, self.weight, bias, self.eps)
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)


class GQACausalSelfAttention(nn.Module):
    """
    Grouped Query Attention with RoPE.

    n_head    : number of query heads
    n_kv_head : number of key/value heads (must divide n_head evenly)
    n_groups  : n_head // n_kv_head  — queries per KV head

    RoPE is applied to Q and K only, with position offset for KV cache decoding.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head    = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_groups  = config.n_head // config.n_kv_head
        self.n_embd    = config.n_embd
        self.head_dim  = config.n_embd // config.n_head
        self.dropout   = config.dropout

        self.q_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                      .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x, rope: Optional[RotaryEmbedding], kv_cache=None):
        B, T, C = x.size()

        # Q: (B, n_head, T, hs)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # K, V: (B, n_kv_head, T, hs)
        kv = self.kv_proj(x).view(B, T, 2, self.n_kv_head, self.head_dim)
        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)   # (B, n_kv_head, T, hs)
        v = v.transpose(1, 2)   # (B, n_kv_head, T, hs)

        # RoPE: apply to Q and K (only when use_rope=True)
        if rope is not None:
            offset = kv_cache[0].size(2) if kv_cache is not None else 0
            cos, sin = rope.get_cos_sin(T, offset=offset, device=x.device)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)

        # Append to KV cache
        is_first_pass = kv_cache is None
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        kv_cache = (k, v)

        # Expand KV heads: (B, n_kv_head, S, hs) -> (B, n_head, S, hs)
        k_expanded = k.repeat_interleave(self.n_groups, dim=1)
        v_expanded = v.repeat_interleave(self.n_groups, dim=1)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_first_pass,
            )
        else:
            att = (q @ k_expanded.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if is_first_pass:
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v_expanded

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, kv_cache


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        if HAS_TRITON and getattr(config, 'use_triton', False):
            self.gelu = TritonGELU()
        self.use_triton = HAS_TRITON and getattr(config, 'use_triton', False)

    def forward(self, x):
        if self.use_triton and x.is_cuda and self.c_fc.bias is not None:
            x = F.linear(x, self.c_fc.weight)
            x = triton_gelu_bias(x, self.c_fc.bias)
        else:
            x = self.c_fc(x)
            x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class BlockGQA(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias, use_triton=getattr(config, 'use_triton', False))
        self.attn = GQACausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias, use_triton=getattr(config, 'use_triton', False))
        self.mlp  = MLP(config)

    def forward(self, x, rope: RotaryEmbedding, kv_cache=None):
        attn_out, kv_cache = self.attn(self.ln_1(x), rope=rope, kv_cache=kv_cache)

        if HAS_TRITON and self.ln_2.use_triton and x.is_cuda:
            bias = self.ln_2.bias if self.ln_2.bias is not None else torch.zeros_like(self.ln_2.weight)
            x, ln_2_x = triton_layernorm_residual(x, attn_out, self.ln_2.weight, bias, self.ln_2.eps)
        else:
            x = x + attn_out
            ln_2_x = self.ln_2(x)

        x = x + self.mlp(ln_2_x)
        return x, kv_cache


# -----------------------------------------------------------------------------
# Config + Model
# -----------------------------------------------------------------------------

@dataclass
class GPTConfigGQA:
    block_size:  int   = 1024
    vocab_size:  int   = 50304   # padded to nearest multiple of 64
    n_layer:     int   = 12
    n_head:      int   = 12
    n_kv_head:   int   = 4       # must divide n_head; set == n_head for MHA behaviour
    n_embd:      int   = 768
    dropout:     float = 0.0
    bias:        bool  = True
    use_triton:  bool  = False
    rope_base:   int   = 10000   # RoPE frequency base (10000 = original, 500000 = LLaMA 3)
    use_rope:    bool  = True    # False = use learned absolute PE (compatible with GPT-2 pretrained weights)


class GPTGQA(nn.Module):

    def __init__(self, config: GPTConfigGQA):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.n_head % config.n_kv_head == 0, \
            f"n_head ({config.n_head}) must be divisible by n_kv_head ({config.n_kv_head})"
        self.config = config

        head_dim = config.n_embd // config.n_head

        transformer_dict = dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([BlockGQA(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias, use_triton=config.use_triton),
        )
        if config.use_rope:
            transformer_dict['rope'] = RotaryEmbedding(head_dim, config.block_size, base=config.rope_base)
        else:
            transformer_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        # RoPE buffers are not parameters, nothing to subtract there.
        # wte is shared with lm_head; subtract nothing extra (same as original nanoGPT).
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, idx, targets=None, kv_caches=None):
        device = idx.device
        b, t = idx.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Token + position embeddings
        tok_emb = self.transformer.wte(idx)  # (B, T, C)
        if self.config.use_rope:
            x = self.transformer.drop(tok_emb)
            rope = self.transformer.rope
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)  # (T, C)
            x = self.transformer.drop(tok_emb + pos_emb)
            rope = None
        new_kv_caches = []
        for i, block in enumerate(self.transformer.h):
            block_kv_cache = kv_caches[i] if kv_caches is not None else None
            x, new_block_kv_cache = block(x, rope=rope, kv_cache=block_kv_cache)
            new_kv_caches.append(new_block_kv_cache)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_kv_caches

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def crop_block_size(self, block_size):
        """Shrink max sequence length. RoPE cache rebuilds lazily on next forward."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # RoPE will rebuild its cache up to the new block_size on next get_cos_sin call.
        # No wpe to crop.

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        print(f"num decayed parameter tensors: {len(decay_params)}, "
              f"with {sum(p.numel() for p in decay_params):,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, "
              f"with {sum(p.numel() for p in nodecay_params):,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas,
            **(dict(fused=True) if use_fused else {})
        )
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate MFU in units of A100 bfloat16 peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token  = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter   = flops_per_fwdbwd * fwdbwd_per_iter
        return (flops_per_iter / dt) / 312e12

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        if not use_cache:
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
            return idx

        kv_caches = None
        curr_idx = idx
        for i in range(max_new_tokens):
            idx_cond = curr_idx if i == 0 else curr_idx[:, [-1]]
            total_len = (kv_caches[0][0].size(2) if kv_caches is not None else 0) + idx_cond.size(1)
            if total_len > self.config.block_size:
                break
            logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            curr_idx = torch.cat((curr_idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return curr_idx

    # ------------------------------------------------------------------
    # Checkpoint loading from a standard MHA nanoGPT checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_mha_checkpoint(cls, ckpt_path: str, n_kv_head: int, override_args: Optional[dict] = None):
        """
        Load a standard MHA nanoGPT checkpoint and convert to GQA+RoPE.

        Note: the MHA checkpoint used learned wpe; those weights are discarded
        since RoPE has no learned position parameters. Q/K will be rotated
        differently from the original model, so a few fine-tuning steps are
        expected to re-adapt — but the MLP and embedding weights transfer fully.

        Weight mapping per layer (MHA -> GQA):
            transformer.h.N.attn.c_attn  [3*C, C]
              -> q_proj.weight  [C, C]           — Q rows, full copy
              -> kv_proj.weight [2*kv_dim, C]    — K+V rows, first n_kv_head heads
            transformer.h.N.attn.c_proj  [C, C]  — direct copy
            transformer.wpe                       — SKIPPED (RoPE replaces it)
            All other weights (ln, mlp, wte, ln_f) — direct copy.
        """
        override_args = override_args or {}

        print(f"Loading MHA checkpoint from {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        mha_args = checkpoint['model_args']

        config_fields = dict(
            block_size = mha_args['block_size'],
            vocab_size = mha_args['vocab_size'],
            n_layer    = mha_args['n_layer'],
            n_head     = mha_args['n_head'],
            n_kv_head  = n_kv_head,
            n_embd     = mha_args['n_embd'],
            dropout    = mha_args.get('dropout', 0.0),
            bias       = mha_args['bias'],
        )
        config_fields.update(override_args)
        config = GPTConfigGQA(**config_fields)

        assert config.n_head % config.n_kv_head == 0, \
            f"n_head ({config.n_head}) must be divisible by n_kv_head ({config.n_kv_head})"

        model = cls(config)
        gqa_sd = model.state_dict()

        mha_sd = checkpoint['model']
        mha_sd = {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v
                  for k, v in mha_sd.items()}

        head_dim = config.n_embd // config.n_head
        kv_dim   = config.n_kv_head * head_dim

        copied = skipped = 0
        with torch.no_grad():
            for gqa_key in gqa_sd.keys():
                if gqa_key.endswith('attn.q_proj.weight'):
                    mha_key = gqa_key.replace('attn.q_proj.weight', 'attn.c_attn.weight')
                    gqa_sd[gqa_key].copy_(mha_sd[mha_key][:config.n_embd, :])
                    copied += 1

                elif gqa_key.endswith('attn.q_proj.bias'):
                    mha_key = gqa_key.replace('attn.q_proj.bias', 'attn.c_attn.bias')
                    if mha_key in mha_sd:
                        gqa_sd[gqa_key].copy_(mha_sd[mha_key][:config.n_embd])
                        copied += 1

                elif gqa_key.endswith('attn.kv_proj.weight'):
                    mha_key = gqa_key.replace('attn.kv_proj.weight', 'attn.c_attn.weight')
                    k_w = mha_sd[mha_key][config.n_embd           : config.n_embd + kv_dim, :]
                    v_w = mha_sd[mha_key][config.n_embd * 2       : config.n_embd * 2 + kv_dim, :]
                    gqa_sd[gqa_key].copy_(torch.cat([k_w, v_w], dim=0))
                    copied += 1

                elif gqa_key.endswith('attn.kv_proj.bias'):
                    mha_key = gqa_key.replace('attn.kv_proj.bias', 'attn.c_attn.bias')
                    if mha_key in mha_sd:
                        k_b = mha_sd[mha_key][config.n_embd     : config.n_embd + kv_dim]
                        v_b = mha_sd[mha_key][config.n_embd * 2 : config.n_embd * 2 + kv_dim]
                        gqa_sd[gqa_key].copy_(torch.cat([k_b, v_b], dim=0))
                        copied += 1

                else:
                    if gqa_key in mha_sd:
                        assert gqa_sd[gqa_key].shape == mha_sd[gqa_key].shape, \
                            f"Shape mismatch for {gqa_key}: {gqa_sd[gqa_key].shape} vs {mha_sd[gqa_key].shape}"
                        gqa_sd[gqa_key].copy_(mha_sd[gqa_key])
                        copied += 1
                    elif gqa_key == 'transformer.wpe.weight' and 'transformer.wpe.weight' in mha_sd:
                        # use_rope=False: copy absolute PE from MHA checkpoint
                        gqa_sd[gqa_key].copy_(mha_sd['transformer.wpe.weight'])
                        copied += 1
                    else:
                        # RoPE buffers or wpe (when use_rope=True) — expected to be absent
                        skipped += 1

        model.load_state_dict(gqa_sd)
        pe_desc = "absolute PE copied from MHA" if not config.use_rope else "RoPE (no weights to copy)"
        print(f"  Loaded: {copied} tensors copied, {skipped} skipped. PE: {pe_desc}")
        print(f"  MHA n_head={config.n_head} -> GQA n_kv_head={config.n_kv_head} "
              f"(kept first {config.n_kv_head} of {config.n_head} KV heads per layer)")
        return model, checkpoint.get('iter_num', 0), checkpoint.get('best_val_loss', 1e9)
