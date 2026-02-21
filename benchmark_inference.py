
import time
import torch
from model import GPTConfig, GPT

def benchmark_inference(model_type='gpt2'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Increase prompt size and max tokens to see scaling
    prompt_len = 128
    idx = torch.randint(0, 50257, (1, prompt_len), dtype=torch.long, device=device)
    max_new_tokens = 100
    
    # 1. Baseline
    print("\n--- 1. Baseline (No Cache, No Triton) ---")
    model_base = GPT.from_pretrained(model_type)
    model_base.to(device)
    model_base.eval()
    
    # Warmup
    for _ in range(2): _ = model_base.generate(idx, 5, use_cache=False)
    
    torch.cuda.synchronize()
    start = time.time()
    _ = model_base.generate(idx, max_new_tokens, use_cache=False)
    torch.cuda.synchronize()
    t_base = time.time() - start
    print(f"Time: {t_base:.4f}s ({max_new_tokens/t_base:.2f} tok/s)")
    
    # 2. KV Cache
    print("\n--- 2. KV Cache (No Triton) ---")
    # Warmup
    for _ in range(2): _ = model_base.generate(idx, 5, use_cache=True)
    
    torch.cuda.synchronize()
    start = time.time()
    _ = model_base.generate(idx, max_new_tokens, use_cache=True)
    torch.cuda.synchronize()
    t_cache = time.time() - start
    print(f"Time: {t_cache:.4f}s ({max_new_tokens/t_cache:.2f} tok/s)")
    
    # 3. Triton + KV Cache
    print("\n--- 3. Triton + KV Cache ---")
    tmp_model = GPT.from_pretrained(model_type)
    config = tmp_model.config
    config.use_triton = True
    model_triton = GPT(config)
    model_triton.load_state_dict(tmp_model.state_dict())
    model_triton.to(device)
    model_triton.eval()
    
    # Warmup
    for _ in range(2): _ = model_triton.generate(idx, 5, use_cache=True)
    
    torch.cuda.synchronize()
    start = time.time()
    _ = model_triton.generate(idx, max_new_tokens, use_cache=True)
    torch.cuda.synchronize()
    t_triton = time.time() - start
    print(f"Time: {t_triton:.4f}s ({max_new_tokens/t_triton:.2f} tok/s)")
    
    # 4. Optimized Mode (torch.compile + Triton + KV Cache)
    print("\n--- 4. Optimized Mode (torch.compile + Triton + KV Cache) ---")
    # Note: torch.compile takes a while to compile, we benchmark after compilation
    model_opt = torch.compile(model_triton)
    print("Compiling optimized model (this may take a minute)...")
    # Warmup + Trigger compilation
    for _ in range(2): _ = model_opt.generate(idx, 5, use_cache=True)
    
    torch.cuda.synchronize()
    start = time.time()
    _ = model_opt.generate(idx, max_new_tokens, use_cache=True)
    torch.cuda.synchronize()
    t_opt = time.time() - start
    print(f"Time: {t_opt:.4f}s ({max_new_tokens/t_opt:.2f} tok/s)")
    
    print("\n" + "="*40)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Baseline:       {max_new_tokens/t_base:>8.2f} tokens/sec")
    print(f"KV Cache:       {max_new_tokens/t_cache:>8.2f} tokens/sec ({(t_base/t_cache):.1f}x speedup)")
    print(f"Triton + KV:    {max_new_tokens/t_triton:>8.2f} tokens/sec ({(t_base/t_triton):.1f}x speedup)")
    print(f"Optimized Mode: {max_new_tokens/t_opt:>8.2f} tokens/sec ({(t_base/t_opt):.1f}x speedup)")
    print("="*40)

if __name__ == "__main__":
    benchmark_inference()
