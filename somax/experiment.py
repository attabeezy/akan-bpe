"""Experiment orchestration for SOMAX fertility comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from somax.datasets import load_jsonl_samples, samples_to_texts
from somax.metrics import compute_fertility
from somax.tokenizers import load_tokenizer


@dataclass(frozen=True)
class ExperimentTokenizer:
    """Tokenizer reference included in an experiment."""

    name: str
    reference: str


def _winner_key(result_map: dict[str, dict[str, object]], test_set_name: str) -> str:
    return min(result_map, key=lambda key: result_map[key][test_set_name]["fertility"])


def run_fertility_experiment(
    experiment_id: str,
    tokenizers: list[ExperimentTokenizer],
    asr_test_file: str,
    tts_test_file: str,
    max_samples: int | None = None,
) -> dict[str, object]:
    """Run one complete tokenizer fertility experiment and return a JSON-safe payload."""
    asr_samples = load_jsonl_samples(Path(asr_test_file))
    tts_samples = load_jsonl_samples(Path(tts_test_file))
    if max_samples is not None:
        asr_samples = asr_samples[:max_samples]
        tts_samples = tts_samples[:max_samples]

    asr_texts = samples_to_texts(asr_samples)
    tts_texts = samples_to_texts(tts_samples)
    if not asr_texts or not tts_texts:
        raise ValueError("Both ASR and TTS test files must contain at least one valid text sample.")

    results: dict[str, dict[str, object]] = {}
    for tokenizer_ref in tokenizers:
        tokenizer = load_tokenizer(tokenizer_ref.reference)
        asr_result = compute_fertility(
            tokenizer_name=tokenizer_ref.name,
            tokenizer_ref=tokenizer_ref.reference,
            test_set_name="asr_test",
            source_file=asr_test_file,
            texts=asr_texts,
            tokenizer=tokenizer,
        )
        tts_result = compute_fertility(
            tokenizer_name=tokenizer_ref.name,
            tokenizer_ref=tokenizer_ref.reference,
            test_set_name="tts_test",
            source_file=tts_test_file,
            texts=tts_texts,
            tokenizer=tokenizer,
        )
        results[tokenizer_ref.name] = {
            "asr_test": asr_result.to_dict(),
            "tts_test": tts_result.to_dict(),
        }

    return {
        "experiment_id": experiment_id,
        "language": "twi",
        "metric": "fertility",
        "formula": "total_tokens / total_words",
        "test_sets": {
            "asr_test": asr_test_file,
            "tts_test": tts_test_file,
        },
        "tokenizers": {
            tokenizer_ref.name: {"reference": tokenizer_ref.reference}
            for tokenizer_ref in tokenizers
        },
        "results": results,
        "summary": {
            "best_on_asr_test": _winner_key(results, "asr_test"),
            "best_on_tts_test": _winner_key(results, "tts_test"),
        },
    }
