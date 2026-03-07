"""
Triton kernels for Grouped Query Attention (GQA).

Two kernels:
  1. triton_gqa_prefill  — causal flash-attention for training / prefill.
       Q: (B, n_head,    T, head_dim)
       K: (B, n_kv_head, T, head_dim)   ← NOT expanded, native GQA
       V: (B, n_kv_head, T, head_dim)
     Each query block knows its KV-head index = q_head // n_groups.
     Saves memory and compute vs repeat_interleave + SDPA.

  2. triton_gqa_decode   — single decode step with KV cache.
       Q: (B, n_head,    1, head_dim)
       K: (B, n_kv_head, S, head_dim)   S = cache_len + 1
       V: (B, n_kv_head, S, head_dim)
     Each query head attends its one KV head across the full sequence.
     Outputs (B, n_head, 1, head_dim).

Both kernels handle arbitrary n_groups = n_head // n_kv_head.
Supports head_dim in {32, 64, 128} — the common sizes.
"""

import math
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Prefill kernel (causal, full sequence)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _gqa_prefill_kernel(
    Q, K, V, Out,
    sm_scale,
    # Q strides: (batch, q_head, seq, dim)
    stride_qb, stride_qh, stride_qm, stride_qd,
    # K strides: (batch, kv_head, seq, dim)
    stride_kb, stride_kh, stride_kn, stride_kd,
    # V strides: same layout as K
    stride_vb, stride_vh, stride_vn, stride_vd,
    # O strides: same as Q
    stride_ob, stride_oh, stride_om, stride_od,
    B, N_Q_HEAD, N_KV_HEAD, T,
    N_GROUPS: tl.constexpr,   # n_head // n_kv_head
    BLOCK_M:  tl.constexpr,   # query block size (rows)
    BLOCK_N:  tl.constexpr,   # key block size   (cols)
    D_HEAD:   tl.constexpr,
):
    """
    Each program handles one (batch, q_head, BLOCK_M rows of Q).
    The KV head index is q_head // N_GROUPS.
    """
    # ── grid indices ──────────────────────────────────────────────────────────
    start_m  = tl.program_id(0)   # which block of query rows
    off_bh   = tl.program_id(1)   # packed (batch * n_q_head) index
    batch    = off_bh // N_Q_HEAD
    q_head   = off_bh %  N_Q_HEAD
    kv_head  = q_head // N_GROUPS

    # ── row / col offsets ──────────────────────────────────────────────────────
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)   # query rows
    offs_n = tl.arange(0, BLOCK_N)                         # key cols (sliding)
    offs_d = tl.arange(0, D_HEAD)

    # ── Q pointer for this block ───────────────────────────────────────────────
    q_base = Q + batch * stride_qb + q_head * stride_qh
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < T, other=0.0)   # (BLOCK_M, D_HEAD)

    # ── K / V base pointers for this batch + kv_head ─────────────────────────
    k_base = K + batch * stride_kb + kv_head * stride_kh
    v_base = V + batch * stride_vb + kv_head * stride_vh

    # ── online softmax state ──────────────────────────────────────────────────
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # ── causal: only attend to positions <= max(offs_m) ───────────────────────
    causal_bound = (start_m + 1) * BLOCK_M  # exclusive upper bound

    for start_n in range(0, causal_bound, BLOCK_N):
        k_ptrs = k_base + (start_n + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[None, :] < T, other=0.0)  # (D_HEAD, BLOCK_N)

        # qk: (BLOCK_M, BLOCK_N)
        qk = tl.dot(q, k) * sm_scale

        # causal mask: query row i must not attend to key col j > i
        causal_mask = offs_m[:, None] >= (start_n + offs_n)[None, :]
        pad_mask    = (start_n + offs_n)[None, :] < T
        qk = tl.where(causal_mask & pad_mask, qk, float('-inf'))

        # online softmax update
        m_ij    = tl.max(qk, axis=1)
        p       = tl.exp(qk - m_ij[:, None])
        p       = tl.where(m_ij[:, None] == float('-inf'), 0.0, p)
        l_ij    = tl.sum(p, axis=1)

        m_new   = tl.maximum(m_i, m_ij)
        alpha   = tl.exp(m_i  - m_new)
        beta    = tl.exp(m_ij - m_new)
        alpha   = tl.where(m_new == float('-inf'), 0.0, alpha)
        beta    = tl.where(m_new == float('-inf'), 0.0, beta)

        acc     = acc * alpha[:, None]
        l_i     = l_i * alpha + l_ij * beta
        m_i     = m_new

        v_ptrs = v_base + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < T, other=0.0)  # (BLOCK_N, D_HEAD)
        acc    += tl.dot(p.to(v.dtype), v) * beta[:, None]

    # ── normalize and store ───────────────────────────────────────────────────
    denom = tl.where(l_i > 0, l_i, 1.0)
    acc   = acc / denom[:, None]
    acc   = tl.where(l_i[:, None] > 0, acc, 0.0)

    o_base = Out + batch * stride_ob + q_head * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=offs_m[:, None] < T)


def triton_gqa_prefill(q, k, v):
    """
    GQA causal prefill (training / full-sequence forward).

    Args:
        q: (B, n_head,    T, head_dim)  float16 or bfloat16
        k: (B, n_kv_head, T, head_dim)
        v: (B, n_kv_head, T, head_dim)

    Returns:
        out: (B, n_head, T, head_dim)  same dtype as input
    """
    B, n_head, T, head_dim = q.shape
    n_kv_head = k.shape[1]
    assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"
    assert head_dim in {32, 64, 128}, f"head_dim {head_dim} not supported (need 32/64/128)"

    n_groups  = n_head // n_kv_head
    sm_scale  = 1.0 / math.sqrt(head_dim)
    BLOCK_M   = 64
    BLOCK_N   = 32

    # cast to bf16 for compute (kernel stores bf16, cast back to input dtype after)
    orig_dtype = q.dtype
    q = q.to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    out = torch.empty_like(q)

    grid = (triton.cdiv(T, BLOCK_M), B * n_head)

    _gqa_prefill_kernel[grid](
        q, k, v, out,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, n_head, n_kv_head, T,
        N_GROUPS=n_groups,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D_HEAD=head_dim,
        num_warps=4,
        num_stages=2,
    )
    return out.to(orig_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Decode kernel (single new token against KV cache)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _gqa_decode_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_od,
    B, N_Q_HEAD, N_KV_HEAD, S,
    N_GROUPS: tl.constexpr,
    BLOCK_N:  tl.constexpr,
    D_HEAD:   tl.constexpr,
):
    """
    One program per (batch, q_head).
    Q has shape (B, n_head, 1, head_dim) — one token per head.
    K, V have shape (B, n_kv_head, S, head_dim) — full cache.
    No causal mask needed: attending to all S positions (all are in the past).
    """
    off_bh  = tl.program_id(0)
    batch   = off_bh // N_Q_HEAD
    q_head  = off_bh %  N_Q_HEAD
    kv_head = q_head // N_GROUPS

    offs_d = tl.arange(0, D_HEAD)
    offs_n = tl.arange(0, BLOCK_N)

    # load Q (single token): (D_HEAD,)
    q_ptr = Q + batch * stride_qb + q_head * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptr)  # (D_HEAD,)

    k_base = K + batch * stride_kb + kv_head * stride_kh
    v_base = V + batch * stride_vb + kv_head * stride_vh

    # online softmax
    m_i = tl.full([], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([D_HEAD], dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        mask = (start_n + offs_n) < S

        k_ptrs = k_base + (start_n + offs_n)[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)  # (BLOCK_N, D_HEAD)

        # dot q with each key: (BLOCK_N,)
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(mask, qk, float('-inf'))

        m_ij  = tl.max(qk, axis=0)
        p     = tl.exp(qk - m_ij)
        p     = tl.where(mask, p, 0.0)
        l_ij  = tl.sum(p, axis=0)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i  - m_new)
        beta  = tl.exp(m_ij - m_new)
        alpha = tl.where(m_new == float('-inf'), 0.0, alpha)
        beta  = tl.where(m_new == float('-inf'), 0.0, beta)

        acc   = acc * alpha
        l_i   = l_i * alpha + l_ij * beta
        m_i   = m_new

        v_ptrs = v_base + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)  # (BLOCK_N, D_HEAD)
        acc += tl.sum(p[:, None] * v, axis=0) * beta

    denom = tl.where(l_i > 0, l_i, 1.0)
    acc   = acc / denom
    acc   = tl.where(l_i > 0, acc, 0.0)

    o_ptr = Out + batch * stride_ob + q_head * stride_oh + offs_d * stride_od
    tl.store(o_ptr, acc.to(tl.bfloat16))


def triton_gqa_decode(q, k, v):
    """
    GQA decode step: single new token attending to KV cache.

    Args:
        q: (B, n_head,    1, head_dim)
        k: (B, n_kv_head, S, head_dim)   S = full sequence length (cache + 1)
        v: (B, n_kv_head, S, head_dim)

    Returns:
        out: (B, n_head, 1, head_dim)
    """
    B, n_head, _, head_dim = q.shape
    n_kv_head = k.shape[1]
    S         = k.shape[2]
    assert n_head % n_kv_head == 0
    assert head_dim in {32, 64, 128}

    n_groups = n_head // n_kv_head
    sm_scale = 1.0 / math.sqrt(head_dim)
    BLOCK_N  = min(128, triton.next_power_of_2(S))

    orig_dtype = q.dtype
    q = q.to(torch.bfloat16).squeeze(2)   # (B, n_head, head_dim) — remove seq dim
    k = k.to(torch.bfloat16)
    v = v.to(torch.bfloat16)

    out_flat = torch.empty(B, n_head, head_dim, dtype=torch.bfloat16, device=q.device)

    grid = (B * n_head,)

    _gqa_decode_kernel[grid](
        q, k, v, out_flat,
        sm_scale,
        q.stride(0),  q.stride(1),  q.stride(2),
        k.stride(0),  k.stride(1),  k.stride(2),  k.stride(3),
        v.stride(0),  v.stride(1),  v.stride(2),  v.stride(3),
        out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
        B, n_head, n_kv_head, S,
        N_GROUPS=n_groups,
        BLOCK_N=BLOCK_N,
        D_HEAD=head_dim,
        num_warps=4,
        num_stages=1,
    )

    return out_flat.unsqueeze(2).to(orig_dtype)   # (B, n_head, 1, head_dim)
