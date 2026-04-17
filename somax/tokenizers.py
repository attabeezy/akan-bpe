"""Tokenizer training and loading utilities for SOMAX."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


DEFAULT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<s>", "</s>", "<pad>"]


@dataclass(frozen=True)
class TrainedTokenizerInfo:
    """Metadata returned after training a tokenizer."""

    name: str
    output_path: str
    vocab_size: int
    num_texts: int
    special_tokens: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "output_path": self.output_path,
            "vocab_size": self.vocab_size,
            "num_texts": self.num_texts,
            "special_tokens": self.special_tokens,
        }


def train_bpe_tokenizer(
    texts: list[str],
    output_path: Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    name: str = "tokenizer",
) -> TrainedTokenizerInfo:
    """Train and save a BPE tokenizer."""
    if not texts:
        raise ValueError("Tokenizer training requires at least one non-empty text sample.")

    tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=tokens, show_progress=True)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    return TrainedTokenizerInfo(
        name=name,
        output_path=str(output_path),
        vocab_size=len(tokenizer.get_vocab()),
        num_texts=len(texts),
        special_tokens=tokens,
    )


def load_tokenizer(reference: str):
    """Load a tokenizer from a local JSON path or Hugging Face identifier."""
    path = Path(reference)
    if path.exists():
        return PreTrainedTokenizerFast(
            tokenizer_file=str(path),
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="[UNK]",
        )
    tokenizer = AutoTokenizer.from_pretrained(reference)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_tokenizer_stats(info: TrainedTokenizerInfo, texts: list[str]) -> dict[str, object]:
    """Build a small metadata payload for one tokenizer training run."""
    lengths = Counter(len(text.split()) for text in texts)
    return {
        **info.to_dict(),
        "word_count_histogram": dict(sorted(lengths.items())),
    }


def save_tokenizer_stats(path: Path, stats: dict[str, object]) -> None:
    """Save tokenizer metadata as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
