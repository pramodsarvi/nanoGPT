num non-decayed parameter tensors: 25, with 19,200 parameters
using fused AdamW: True
/root/ngpt/nanoGPT/train_fsdp.py:256: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=False)
compiling the model... (takes a ~minute)
W0228 03:48:33.312000 2983905 torch/_inductor/utils.py:1558] [0/0] Not enough SMs to use max_autotune_gemm mode



d=False)
step 0: train loss 4.4534, val loss 4.4421
saving checkpoint to out_experiments/exp01_mha_abspe
iter 0: loss 4.4273, time 113683.25ms

[35] 0:bash*                              why has time increased significantly
113 seconds for iter 0 is the Flash Attention kernel being compiled by PyTorch on first use (even with compile=False, F.scaled_dot_product_attention JIT-compiles its kernels the first time). Subsequent iterations will be much faster.

Check iter 1 and 2 — they should drop to ~100-300ms range. The first iter always includes this one-time kernel compilation cost.


