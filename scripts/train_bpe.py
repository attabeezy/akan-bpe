#!/usr/bin/env python3
"""Train BPE vocabularies for ASR and TTS streams.

Separate vocabularies are trained for:
- ASR stream: Noisy, conversational, code-switching text
- TTS stream: Clean, formal, grammatically correct text

Usage:
    python scripts/01_train_bpe.py --input data/akan/ --output models/tokenizers/ --vocab-size 8000
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Iterator

from tokenizers import Trie, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer


def read_jsonl_texts(jsonl_path: Path) -> Iterator[str]:
    """Extract transcription/text from JSONL file.

    Args:
        jsonl_path: Path to JSONL file with 'transcription' or 'text' field.

    Yields:
        Text content from each line.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                text = item.get("transcription") or item.get("text", "")
                if text:
                    yield text
            except json.JSONDecodeError:
                continue


def train_bpe_tokenizer(
    texts: Iterator[str], vocab_size: int, output_path: Path, language: str, stream_type: str
) -> dict:
    """Train a BPE tokenizer on the given texts.

    Args:
        texts: Iterator of text samples.
        vocab_size: Target vocabulary size.
        output_path: Path to save tokenizer JSON.
        language: Language code for metadata.
        stream_type: 'asr' or 'tts' for metadata.

    Returns:
        Training statistics.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        show_progress=True,
    )

    texts_list = list(texts)
    print(f"Training on {len(texts_list)} samples...")

    tokenizer.train_from_iterator(texts_list, trainer=trainer)

    tokenizer.save(str(output_path))

    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")

    return {
        "vocab_size": len(vocab),
        "num_samples": len(texts_list),
        "language": language,
        "stream_type": stream_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE vocabularies")
    parser.add_argument("--input", type=str, default="data/akan/")
    parser.add_argument("--output", type=str, default="models/tokenizers/")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--language", type=str, default="akan")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) / args.language
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    print(f"Training BPE vocabularies for {args.language}...")
    print(f"  Input: {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Vocab size: {args.vocab_size}")
    print()

    stats = {}

    for stream_type in ["asr", "tts"]:
        config_name = (
            f"{args.language.split('_')[0]}_{stream_type}" if stream_type == "asr" else None
        )
        if stream_type == "tts":
            if args.language == "akan":
                pattern = "twi_tts"
            else:
                pattern = f"{args.language}_tts"
        else:
            pattern = (
                f"{args.language.split('_')[0]}_asr"
                if "_" in args.language
                else f"{args.language}_asr"
            )

        train_file = input_dir / f"{pattern}_train.jsonl"
        if not train_file.exists():
            print(f"WARNING: {train_file} not found, skipping {stream_type.upper()}")
            continue

        print(f"--- Training {stream_type.upper()} tokenizer ---")

        output_file = output_dir / f"{stream_type}_tokenizer.json"
        stream_stats = train_bpe_tokenizer(
            texts=read_jsonl_texts(train_file),
            vocab_size=args.vocab_size,
            output_path=output_file,
            language=args.language,
            stream_type=stream_type,
        )
        stats[stream_type] = stream_stats
        print(f"Saved to: {output_file}")
        print()

    stats_file = output_dir / "training_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Training statistics saved to: {stats_file}")
    print("Training complete!")


if __name__ == "__main__":
    main()
