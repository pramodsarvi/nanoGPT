
import torch
from model import GPTConfig, GPT
import triton_kernels

def check_correctness():
    device = 'cuda'
    B, H, T, D = 2, 4, 128, 64
    dtype = torch.float16
    
    q = torch.randn((B, H, T, D), device=device, dtype=dtype)
    k = torch.randn((B, H, T, D), device=device, dtype=dtype)
    v = torch.randn((B, H, T, D), device=device, dtype=dtype)
    
    # Reference
    res_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    
    # Triton
    res_triton = triton_kernels.triton_attention(q, k, v, causal=True)
    
    diff = (res_ref - res_triton).abs().max()
    print(f"Max difference between PyTorch and Triton Attention: {diff:.6f}")
    
    if diff < 1e-2:
        print("Correctness check passed!")
    else:
        print("Correctness check failed!")

if __name__ == "__main__":
    check_correctness()
