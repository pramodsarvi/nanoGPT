# Streams FineWeb-Edu sample-10BT from HuggingFace without downloading the full dataset.
# Uses tiktoken GPT-2 BPE tokenizer (vocab 50257, fits in uint16).
#
# Usage:
#   python data/fineweb_edu/prepare.py                        # full 10BT (~19GB)
#   python data/fineweb_edu/prepare.py --max_tokens 1_000_000_000  # 1BT (~2GB)
#   python data/fineweb_edu/prepare.py --max_tokens 100_000_000    # 100MT (~200MB)
#
# Output:
#   data/fineweb_edu/train.bin  (size depends on --max_tokens)
#   data/fineweb_edu/val.bin    (~8MB)
#   data/fineweb_edu/meta.pkl

import os
import pickle
import argparse
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

# ~10B tokens total in sample-10BT
# We reserve the first VAL_DOCS documents for validation
VAL_DOCS = 5000
SHARD_SIZE = 1024 * 1024 * 64  # flush to disk every 64M tokens to bound RAM usage

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256

data_dir = os.path.dirname(__file__)
train_bin = os.path.join(data_dir, 'train.bin')
val_bin   = os.path.join(data_dir, 'val.bin')

def tokenize(text):
    ids = enc.encode_ordinary(text)
    ids.append(EOT)
    return np.array(ids, dtype=np.uint16)

def write_bin(path, token_buffer):
    arr = np.array(token_buffer, dtype=np.uint16)
    with open(path, 'ab') as f:
        arr.tofile(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='Max train tokens to write (default: all ~10B). '
                             'Examples: 100_000_000 (~200MB), 1_000_000_000 (~2GB)')
    parser.add_argument('--val_only', action='store_true',
                        help='Only prepare val.bin (~8MB). Use with hf: streaming training.')
    args = parser.parse_args()

    # Remove stale files
    if args.val_only:
        if os.path.exists(val_bin):
            os.remove(val_bin)
        print("Preparing val.bin only (first 5000 docs ~8MB) ...")
    else:
        for p in [train_bin, val_bin]:
            if os.path.exists(p):
                os.remove(p)
        if args.max_tokens:
            print(f"Streaming HuggingFaceFW/fineweb-edu sample-10BT (capped at {args.max_tokens:,} train tokens) ...")
        else:
            print("Streaming HuggingFaceFW/fineweb-edu sample-10BT (full ~10B tokens) ...")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    train_tokens = 0
    val_tokens   = 0
    train_buf    = []
    val_buf      = []
    doc_idx      = 0

    for example in tqdm(dataset, desc="tokenizing"):
        ids = tokenize(example['text'])

        if doc_idx < VAL_DOCS:
            val_buf.extend(ids.tolist())
            val_tokens += len(ids)
        else:
            train_buf.extend(ids.tolist())
            train_tokens += len(ids)

            if len(train_buf) >= SHARD_SIZE:
                write_bin(train_bin, train_buf)
                train_buf = []

            if args.max_tokens and train_tokens >= args.max_tokens:
                break

        doc_idx += 1

        if args.val_only and doc_idx >= VAL_DOCS:
            break

    # Flush remaining
    if train_buf:
        write_bin(train_bin, train_buf)
    if val_buf:
        write_bin(val_bin, val_buf)

    print(f"\nDone.")
    print(f"  train: {train_tokens:,} tokens -> {train_bin}  ({train_tokens*2/1e9:.2f} GB)")
    print(f"  val:   {val_tokens:,} tokens -> {val_bin}")

    # Save meta for train_fsdp.py vocab size discovery
    meta = {
        'vocab_size': enc.n_vocab,  # 50257
        'tokenizer': 'gpt2',
    }
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"  meta:  vocab_size={meta['vocab_size']} -> {meta_path}")
