"""
Step 2: Tokenize Kannada corpus into train.bin / val.bin using the BPE
tokenizer trained by prepare_tokenizer.py.

Run AFTER prepare_tokenizer.py has completed.

Output:
  data/kannada/train.bin   — uint16 token ids, ~1.5B tokens (Wikipedia + Sangraha)
  data/kannada/val.bin     — uint16 token ids, first VAL_DOCS documents
  data/kannada/meta.pkl    — vocab_size (updated with token counts)

Usage:
  python data/kannada/prepare_data.py                         # full corpus
  python data/kannada/prepare_data.py --max_tokens 500_000_000  # 500M token subset
  python data/kannada/prepare_data.py --no_sangraha           # Wikipedia only (~100M tokens)

Requirements:
  pip install tokenizers datasets numpy tqdm
"""

import os
import argparse
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer

DATA_DIR  = Path(__file__).parent
TOK_PATH  = DATA_DIR / "tokenizer" / "tokenizer.json"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN   = DATA_DIR / "val.bin"
META_PATH = DATA_DIR / "meta.pkl"

VAL_DOCS   = 2000       # first N documents → val set
SHARD_SIZE = 64 * 1024 * 1024  # flush to disk every 64M tokens (~128MB)


def load_tokenizer() -> Tokenizer:
    if not TOK_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOK_PATH}\n"
            "Run prepare_tokenizer.py first:\n"
            "  python data/kannada/prepare_tokenizer.py"
        )
    tok = Tokenizer.from_file(str(TOK_PATH))
    # Enable padding/truncation off — we handle length ourselves
    tok.no_padding()
    tok.no_truncation()
    print(f"Loaded tokenizer: vocab_size={tok.get_vocab_size():,}  path={TOK_PATH}")
    return tok


def tokenize_batch(tok: Tokenizer, texts: list[str], eot_id: int) -> list[int]:
    """Tokenize a batch of texts, appending EOT after each document."""
    ids = []
    encodings = tok.encode_batch(texts)
    for enc in encodings:
        ids.extend(enc.ids)
        ids.append(eot_id)
    return ids


def write_bin(path: Path, token_buffer: list[int]):
    """Append token buffer to a binary file as uint16."""
    arr = np.array(token_buffer, dtype=np.uint16)
    with open(path, "ab") as f:
        arr.tofile(f)


def iter_corpus(use_sangraha: bool, max_sangraha_docs: int):
    """Yield (text, source) pairs from Wikipedia + optionally Sangraha."""
    from datasets import load_dataset

    # ── Wikipedia Kannada ─────────────────────────────────────────────────────
    print("Streaming Wikipedia Kannada ...")
    wiki = load_dataset(
        "wikimedia/wikipedia",
        "20231101.kn",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    for doc in wiki:
        text = doc.get("text", "").strip()
        if text:
            yield text

    # ── Sangraha Kannada ──────────────────────────────────────────────────────
    if use_sangraha:
        print("Streaming ai4bharat/sangraha Kannada ...")
        try:
            sangraha = load_dataset(
                "ai4bharat/sangraha",
                "kn",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            n = 0
            for doc in sangraha:
                text = doc.get("text", "").strip()
                if text and len(text) > 50:
                    yield text
                    n += 1
                if max_sangraha_docs and n >= max_sangraha_docs:
                    break
        except Exception as e:
            print(f"WARNING: Could not load Sangraha ({e}). Skipping.")


def prepare(use_sangraha: bool, max_tokens: int, max_sangraha_docs: int,
            batch_size: int = 512):

    tok    = load_tokenizer()
    eot_id = tok.token_to_id("<|endoftext|>")
    assert eot_id is not None, "tokenizer missing <|endoftext|> special token"

    # Remove stale output files
    for p in [TRAIN_BIN, VAL_BIN]:
        if p.exists():
            p.unlink()
            print(f"Removed stale {p}")

    train_tokens = 0
    val_tokens   = 0
    train_buf    = []
    val_buf      = []
    doc_idx      = 0
    text_batch   = []

    corpus = iter_corpus(use_sangraha, max_sangraha_docs)
    pbar   = tqdm(desc="tokenizing", unit="doc")

    def flush_batch():
        nonlocal doc_idx, train_tokens, val_tokens, train_buf, val_buf
        ids = tokenize_batch(tok, text_batch, eot_id)

        # Route to val or train based on doc_idx at start of batch
        # (approximate: all docs in a batch go to same split)
        if doc_idx < VAL_DOCS:
            val_buf.extend(ids)
            val_tokens += len(ids)
        else:
            train_buf.extend(ids)
            train_tokens += len(ids)

            if len(train_buf) >= SHARD_SIZE:
                write_bin(TRAIN_BIN, train_buf)
                train_buf.clear()

        text_batch.clear()

    for text in corpus:
        text_batch.append(text)
        doc_idx += 1
        pbar.update(1)
        pbar.set_postfix({"train_tok": f"{train_tokens/1e6:.1f}M",
                          "val_tok":   f"{val_tokens/1e6:.1f}M"})

        if len(text_batch) >= batch_size:
            flush_batch()

        if max_tokens and train_tokens >= max_tokens:
            if text_batch:
                flush_batch()
            break

    # Flush remaining
    if text_batch:
        flush_batch()
    if train_buf:
        write_bin(TRAIN_BIN, train_buf)
    if val_buf:
        write_bin(VAL_BIN, val_buf)

    pbar.close()

    # ── report ────────────────────────────────────────────────────────────────
    print(f"\nDone.")
    train_size_gb = train_tokens * 2 / 1e9
    val_size_gb   = val_tokens   * 2 / 1e9
    print(f"  train: {train_tokens:,} tokens → {TRAIN_BIN}  ({train_size_gb:.2f} GB)")
    print(f"  val:   {val_tokens:,}   tokens → {VAL_BIN}   ({val_size_gb:.3f} GB)")

    # Update meta.pkl with token counts
    meta = {}
    if META_PATH.exists():
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
    meta["train_tokens"] = train_tokens
    meta["val_tokens"]   = val_tokens
    meta["eot_token"]    = eot_id
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"  meta:  vocab_size={meta['vocab_size']:,} → {META_PATH}")

    print(f"\nNext step:")
    print(f"  torchrun --standalone --nproc_per_node=<N> train_fsdp.py \\")
    print(f"      config/experiments/exp20_kannada_350m.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize Kannada corpus")
    parser.add_argument("--max_tokens",         type=int, default=0,
                        help="Cap train tokens (0 = all). E.g. 500_000_000 for 500M")
    parser.add_argument("--no_sangraha",        action="store_true",
                        help="Use Wikipedia only (~100M tokens, faster)")
    parser.add_argument("--max_sangraha_docs",  type=int, default=0,
                        help="Cap Sangraha docs (0 = all)")
    parser.add_argument("--batch_size",         type=int, default=512,
                        help="Tokenization batch size (default: 512)")
    args = parser.parse_args()

    prepare(
        use_sangraha=not args.no_sangraha,
        max_tokens=args.max_tokens,
        max_sangraha_docs=args.max_sangraha_docs,
        batch_size=args.batch_size,
    )
