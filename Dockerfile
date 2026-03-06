FROM nvcr.io/nvidia/pytorch:24.12-py3

WORKDIR /workspace

# Clone the repo
RUN git clone --branch gqa-implementation https://github.com/pramodsarvi/nanoGPT.git

WORKDIR /workspace/nanoGPT

# Install Python dependencies not included in NGC image
RUN pip3 install -r requirements.txt

CMD ["python", "-c", "print('nanoGPT container ready.\\n\\nUsage:\\n  torchrun --standalone --nproc_per_node=NUM_GPUS train_fsdp.py CONFIG_FILE\\n\\nExample:\\n  torchrun --standalone --nproc_per_node=2 train_fsdp.py config/experiments/exp15_gqa4_swiglu_lr1e3_fineweb_4gpu.py')"]
