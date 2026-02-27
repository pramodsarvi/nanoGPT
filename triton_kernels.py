
import math
import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_residual_kernel(
    X,      # Pointer to input
    Res,    # Pointer to residual
    Y,      # Pointer to normalized output
    SumOut, # Pointer to sum (X + Res)
    W, B, M, V,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Res += row * stride
    Y += row * stride
    SumOut += row * stride
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load input and residual
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Res + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Fused Add
    x_sum = x + res
    # Store the sum back if needed (or we can just work with it)
    tl.store(SumOut + cols, x_sum, mask=mask)
    
    # Standard LayerNorm on the sum
    mean = tl.sum(x_sum, axis=0) / N
    x_zm = tl.where(mask, x_sum - mean, 0.0)
    var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Optional: store mean/rstd if needed for training (not for inference)
    
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = x_zm * rstd * w + b
    
    tl.store(Y + cols, y, mask=mask)

def triton_layernorm_residual(x, residual, weight, bias, eps):
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    residual = residual.view(-1, x_shape[-1])
    M, N = x.shape
    
    y = torch.empty_like(x)
    x_sum = torch.empty_like(x)
    
    mean = torch.empty((M, ), dtype=torch.float32, device='cuda')
    rstd = torch.empty((M, ), dtype=torch.float32, device='cuda')
    
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK_SIZE >= 2048: num_warps = 8
    
    layernorm_residual_kernel[(M, )](
        x, residual, y, x_sum, weight, bias, mean, rstd,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return x_sum.view(*x_shape), y.view(*x_shape)

@triton.jit
def layernorm_kernel(
    X, Y, W, B, M, V,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_zm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y = x_zm * rstd * w + b
    tl.store(Y + cols, y, mask=mask)

def triton_layernorm(x, weight, bias, eps):
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M, ), dtype=torch.float32, device='cuda')
    rstd = torch.empty((M, ), dtype=torch.float32, device='cuda')
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    layernorm_kernel[(M, )](
        x, y, weight, bias, mean, rstd,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y.view(*x_shape)

@triton.jit
def gelu_kernel(
    x_ptr, y_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.70710678118))
    tl.store(y_ptr + offsets, y.to(tl.float16), mask=mask)

def triton_gelu(x):
    n_elements = x.numel()
    y = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    gelu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y

@triton.jit
def layernorm_gelu_kernel(
    X, Y, W, B,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Load and normalize
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_zm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    b = tl.load(B + cols, mask=mask).to(tl.float32)
    y_norm = x_zm * rstd * w + b
    
    # Apply GELU (Approximate)
    y = 0.5 * y_norm * (1.0 + tl.math.erf(y_norm * 0.70710678118))
    
    tl.store(Y + cols, y.to(tl.float16), mask=mask)

def triton_layernorm_gelu(x, weight, bias, eps):
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    layernorm_gelu_kernel[(M, )](
        x, y, weight, bias,
        x.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return y.view(*x_shape)

@triton.jit
def gelu_bias_kernel(
    x_ptr, bias_ptr, y_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * N
    y_ptr += row * N
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x += bias
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.70710678118))
    tl.store(y_ptr + cols, y.to(tl.float16), mask=mask)

def triton_gelu_bias(x, bias):
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    gelu_bias_kernel[(M,)](x, bias, y, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return y.view(*x_shape)

class TritonLayerNorm(torch.nn.Module):
    def __init__(self, ndim, bias, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(ndim))
        self.bias = torch.nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        if not x.is_cuda:
            return torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)
        
        # If bias is None, we need to pass a zero tensor to the kernel (or modify kernel)
        bias = self.bias if self.bias is not None else torch.zeros_like(self.weight)
        return triton_layernorm(x, self.weight, bias, self.eps)

class TritonLayerNormGELU(torch.nn.Module):
    def __init__(self, ndim, bias, eps=1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(ndim))
        self.bias = torch.nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x):
        if not x.is_cuda:
            y = torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)
            return torch.nn.functional.gelu(y)
        
        bias = self.bias if self.bias is not None else torch.zeros_like(self.weight)
        return triton_layernorm_gelu(x, self.weight, bias, self.eps)

class TritonGELU(torch.nn.Module):
    def forward(self, x):
        if not x.is_cuda:
            return torch.nn.functional.gelu(x)
        return triton_gelu(x)

@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale,
    L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    # -- grid id --
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)

    # batch and head index
    batch_idx = off_hz // H
    head_idx = off_hz % H

    # load Q
    curr_q_ptr = Q + batch_idx * stride_qz + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(curr_q_ptr, mask=offs_m[:, None] < N_CTX, other=0.0)

    # initialize L and M
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    d_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # iterate over K, V
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # load K
        curr_k_ptr = K + batch_idx * stride_kz + head_idx * stride_kh + (start_n + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k = tl.load(curr_k_ptr, mask=(start_n + offs_n)[None, :] < N_CTX, other=0.0)
        # compute qk
        qk = tl.dot(q, k)
        qk *= sm_scale
        # causal mask + padding mask
        mask = (offs_m[:, None] >= (start_n + offs_n)[None, :]) & ((start_n + offs_n)[None, :] < N_CTX)
        qk += tl.where(mask, 0, float("-inf"))
        # softmax
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        l_ij = tl.sum(p, 1)
        # update acc
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        # Correctly handle the case where m_i_new is -inf (all elements masked)
        # In that case, alpha and beta would be NaN or improper.
        # But if m_i_new is -inf, then alpha and beta don't matter because acc will be 0.
        # However, to avoid NaN, we can use 0 where m_i_new is -inf.
        alpha = tl.where(m_i_new == float("-inf"), 0.0, alpha)
        beta = tl.where(m_i_new == float("-inf"), 0.0, beta)
        
        acc = acc * alpha[:, None]
        # load V
        curr_v_ptr = V + batch_idx * stride_vz + head_idx * stride_vh + (start_n + offs_n)[:, None] * stride_vk + offs_d[None, :] * stride_vn
        v = tl.load(curr_v_ptr, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(tl.float16), v.to(tl.float16)) * beta[:, None]
        d_i = d_i * alpha + l_ij * beta
        m_i = m_i_new

    # write output
    acc = acc / tl.where(d_i[:, None] > 0, d_i[:, None], 1.0)
    acc = tl.where(d_i[:, None] > 0, acc, 0.0)
    # mask out rows that are purely padding
    acc = tl.where(offs_m[:, None] < N_CTX, acc, 0.0)
    
    curr_o_ptr = Out + batch_idx * stride_oz + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(curr_o_ptr, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)

def triton_attention(q, k, v, causal=True):
    # shape: (batch, n_heads, seq_len, head_dim)
    q = q.to(torch.float16)
    k = k.to(torch.float16)
    v = v.to(torch.float16)
    sm_scale = 1.0 / math.sqrt(q.size(-1))
    batch, n_heads, seq_len, head_dim = q.shape

    BLOCK_M = 64
    BLOCK_N = 32
    # we only support head_dim that is power of 2
    assert head_dim in {16, 32, 64, 128}

    o = torch.empty_like(q)
    l = torch.empty((batch * n_heads, seq_len), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_heads)

    _attn_fwd_kernel[grid](
        q, k, v, sm_scale,
        l,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, n_heads, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        D_HEAD=head_dim,
        num_warps=4,
        num_stages=2,
    )
    return o
