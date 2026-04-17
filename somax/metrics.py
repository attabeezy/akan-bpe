"""Metrics used by SOMAX experiments."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class FertilityResult:
    """Fertility summary for one tokenizer on one test set."""

    tokenizer_name: str
    tokenizer_ref: str
    test_set_name: str
    source_file: str
    fertility: float
    total_tokens: int
    total_words: int
    num_samples: int
    fertility_std: float

    def to_dict(self) -> dict[str, object]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_ref": self.tokenizer_ref,
            "test_set_name": self.test_set_name,
            "source_file": self.source_file,
            "fertility": self.fertility,
            "total_tokens": self.total_tokens,
            "total_words": self.total_words,
            "num_samples": self.num_samples,
            "fertility_std": self.fertility_std,
        }


def compute_fertility(
    tokenizer_name: str,
    tokenizer_ref: str,
    test_set_name: str,
    source_file: str,
    texts: list[str],
    tokenizer,
) -> FertilityResult:
    """Compute token fertility for one tokenizer over one dataset."""
    total_tokens = 0
    total_words = 0
    per_sample: list[float] = []

    for text in texts:
        words = len(text.split())
        token_count = len(tokenizer.encode(text))
        total_tokens += token_count
        total_words += words
        if words > 0:
            per_sample.append(token_count / words)

    fertility = total_tokens / total_words if total_words else 0.0
    std = statistics.stdev(per_sample) if len(per_sample) > 1 else 0.0
    return FertilityResult(
        tokenizer_name=tokenizer_name,
        tokenizer_ref=tokenizer_ref,
        test_set_name=test_set_name,
        source_file=source_file,
        fertility=fertility,
        total_tokens=total_tokens,
        total_words=total_words,
        num_samples=len(texts),
        fertility_std=std,
    )
