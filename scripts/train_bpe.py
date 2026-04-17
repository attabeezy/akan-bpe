#!/usr/bin/env python3
"""Train one BPE tokenizer for a SOMAX variant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from somax.datasets import load_jsonl_samples, samples_to_texts
from somax.tokenizers import build_tokenizer_stats, save_tokenizer_stats, train_bpe_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one SOMAX BPE tokenizer.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more JSONL input files.")
    parser.add_argument("--output", required=True, help="Tokenizer JSON output path.")
    parser.add_argument("--name", required=True, help="Logical tokenizer name, e.g. asr.")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Target vocabulary size.")
    args = parser.parse_args()

    all_samples = []
    for input_ref in args.inputs:
        path = Path(input_ref)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        all_samples.extend(load_jsonl_samples(path))

    texts = samples_to_texts(all_samples)
    output_path = Path(args.output)
    info = train_bpe_tokenizer(texts=texts, output_path=output_path, vocab_size=args.vocab_size, name=args.name)
    stats = build_tokenizer_stats(info, texts)
    stats_path = output_path.with_name(f"{output_path.stem}_stats.json")
    save_tokenizer_stats(stats_path, stats)

    print(f"Tokenizer saved to {output_path}")
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
