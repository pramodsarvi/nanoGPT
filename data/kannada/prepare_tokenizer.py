"""
Step 1: Train a BPE tokenizer from scratch on Kannada text.

Sources (streamed from HuggingFace, no full download needed):
  - wikimedia/wikipedia  'kn' split  (~200MB text, clean encyclopedic Kannada)
  - ai4bharat/sangraha   'kn' split  (~2-5GB text, broad Kannada web text)

Output (saved to data/kannada/):
  tokenizer/vocab.json      — BPE vocabulary (token → id)
  tokenizer/merges.txt      — BPE merge rules
  tokenizer/tokenizer.json  — full HuggingFace tokenizer (for easy reuse)
  meta.pkl                  — vocab_size for train_fsdp.py

Usage:
  python data/kannada/prepare_tokenizer.py                  # 16K vocab (recommended)
  python data/kannada/prepare_tokenizer.py --vocab_size 32000  # bilingual Kannada+English
  python data/kannada/prepare_tokenizer.py --no_sangraha    # Wikipedia only (faster, ~10 min)

Requirements:
  pip install tokenizers datasets

Time estimate:
  Wikipedia only  : ~10 minutes
  Wikipedia + Sangraha: ~40-60 minutes depending on bandwidth
"""

import os
import argparse
import pickle
import itertools
from pathlib import Path

DATA_DIR = Path(__file__).parent

# ── special tokens ────────────────────────────────────────────────────────────
# Keep consistent with prepare_data.py and training
SPECIAL_TOKENS = [
    "<|endoftext|>",   # id 0 — document separator (same role as GPT-2 EOT)
    "<|pad|>",         # id 1 — padding (masked in loss)
    "<|unk|>",         # id 2 — unknown (rarely used with BPE but good to have)
]


def iter_kannada_text(use_sangraha: bool, max_wiki_docs: int, max_sangraha_docs: int):
    """
    Generator that yields raw Kannada text strings.
    Streams from HuggingFace — no full download.
    """
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
    n = 0
    for doc in wiki:
        text = doc.get("text", "").strip()
        if text:
            yield text
            n += 1
        if max_wiki_docs and n >= max_wiki_docs:
            break
    print(f"  Wikipedia: {n:,} documents")

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
                if text and len(text) > 50:  # skip very short fragments
                    yield text
                    n += 1
                if max_sangraha_docs and n >= max_sangraha_docs:
                    break
            print(f"  Sangraha:  {n:,} documents")
        except Exception as e:
            print(f"  WARNING: Could not load Sangraha ({e}). Using Wikipedia only.")


def train_tokenizer(vocab_size: int, use_sangraha: bool,
                    max_wiki_docs: int, max_sangraha_docs: int):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Sequence, Whitespace, Punctuation, Digits
    from tokenizers.normalizers import NFC
    from tokenizers.decoders import BPEDecoder

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # NFC normalisation — critical for Kannada Unicode (composed form)
    tokenizer.normalizer = NFC()

    # Pre-tokenizer: split on whitespace + punctuation + digits
    # This ensures Kannada aksharas (syllable clusters) are not split from
    # punctuation, but numbers and symbols are always their own tokens.
    tokenizer.pre_tokenizer = Sequence([
        Whitespace(),
        Punctuation(),
        Digits(individual_digits=False),
    ])

    tokenizer.decoder = BPEDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,           # merge pairs seen < 2 times are ignored
        show_progress=True,
        initial_alphabet=[],       # let BPE discover the alphabet from data
    )

    print(f"\nTraining BPE tokenizer (vocab_size={vocab_size:,}) ...")
    print("This may take 10-60 minutes depending on corpus size.\n")

    text_iter = iter_kannada_text(use_sangraha, max_wiki_docs, max_sangraha_docs)
    tokenizer.train_from_iterator(text_iter, trainer=trainer, length=None)

    # ── save ──────────────────────────────────────────────────────────────────
    tok_dir = DATA_DIR / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)

    # Full HF tokenizer JSON (for easy reuse / inspection)
    tok_json = str(tok_dir / "tokenizer.json")
    tokenizer.save(tok_json)
    print(f"\nSaved tokenizer → {tok_json}")

    # Also save vocab + merges separately for reference
    model = tokenizer.model
    model.save(str(tok_dir))  # writes vocab.json + merges.txt
    print(f"Saved vocab.json + merges.txt → {tok_dir}/")

    # meta.pkl — read by train_fsdp.py to set vocab_size in model config
    actual_vocab_size = tokenizer.get_vocab_size()
    meta = {
        "vocab_size": actual_vocab_size,
        "tokenizer": "kannada_bpe",
        "tokenizer_path": str(tok_dir / "tokenizer.json"),
        "special_tokens": {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS},
        "eot_token": tokenizer.token_to_id("<|endoftext|>"),
    }
    meta_path = DATA_DIR / "meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved meta.pkl  → vocab_size={actual_vocab_size:,}")

    # Quick sanity check
    print("\n── Sanity check ──────────────────────────────────────────────────")
    test_sentences = [
        "ಕನ್ನಡ ಭಾಷೆ ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆ.",   # Kannada is the official language of Karnataka state.
        "ಭಾರತ ಒಂದು ಮಹಾನ್ ದೇಶ.",                         # India is a great country.
        "ನಮ್ಮ ದೇಶದ ಸಂಸ್ಕೃತಿ ಬಹಳ ಶ್ರೀಮಂತವಾಗಿದೆ.",        # Our country's culture is very rich.
    ]
    for s in test_sentences:
        enc = tokenizer.encode(s)
        dec = tokenizer.decode(enc.ids)
        print(f"  Input : {s}")
        print(f"  Tokens: {enc.tokens}")
        print(f"  IDs   : {enc.ids}")
        print(f"  Decode: {dec}")
        print()

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Kannada BPE tokenizer")
    parser.add_argument("--vocab_size",         type=int,  default=16000,
                        help="BPE vocabulary size (default: 16000)")
    parser.add_argument("--no_sangraha",        action="store_true",
                        help="Skip Sangraha, use Wikipedia only (faster)")
    parser.add_argument("--max_wiki_docs",      type=int,  default=0,
                        help="Cap Wikipedia docs (0 = all, ~67K docs)")
    parser.add_argument("--max_sangraha_docs",  type=int,  default=0,
                        help="Cap Sangraha docs (0 = all, ~2M docs)")
    args = parser.parse_args()

    train_tokenizer(
        vocab_size=args.vocab_size,
        use_sangraha=not args.no_sangraha,
        max_wiki_docs=args.max_wiki_docs,
        max_sangraha_docs=args.max_sangraha_docs,
    )

    print("\nNext step:")
    print("  python data/kannada/prepare_data.py")
