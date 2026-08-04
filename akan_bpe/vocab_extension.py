"""Build and audit vocabulary-extension tokenizers from a frozen Akan BPE vocabulary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tokenizers import AddedToken
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from akan_bpe.datasets import load_jsonl_samples, samples_to_texts
from akan_bpe.revision_manifest import sha256_file
from akan_bpe.tokenizers import DEFAULT_SPECIAL_TOKENS
from akan_bpe.vocab_ablation import (
    evaluate_tokenizer,
    paired_bootstrap_fertility_difference,
)

EXTENSION_SCHEMA_VERSION = 1
DEFAULT_BASE_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
DEFAULT_SOURCE_TOKENIZER = Path("models/revision_v2/mixed_bpe_v32000.json")
DEFAULT_OUTPUT_DIR = Path("models/revision_v2/qwen3_0.6b_extension_v32000")
DEFAULT_METADATA_OUTPUT = Path("results/vocab_extension_qwen3_0.6b_v32000.json")


def _vocab_digest(vocab: dict[str, int]) -> str:
    canonical = json.dumps(
        sorted(vocab.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_candidate_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    """Load the frozen replacement tokenizer with its declared special-token semantics."""
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="[UNK]",
    )


def select_extension_tokens(
    *,
    candidate_tokenizer: PreTrainedTokenizerBase,
    base_tokenizer: PreTrainedTokenizerBase,
    excluded_special_tokens: set[str] | None = None,
) -> dict[str, Any]:
    """Select every novel non-special candidate in learned vocabulary-ID order."""
    candidate_vocab = candidate_tokenizer.get_vocab()
    base_vocab = base_tokenizer.get_vocab()
    excluded = set(DEFAULT_SPECIAL_TOKENS)
    excluded.update(candidate_tokenizer.all_special_tokens)
    if excluded_special_tokens is not None:
        excluded.update(excluded_special_tokens)

    ordered_candidates = sorted(candidate_vocab.items(), key=lambda item: item[1])
    special_tokens = [token for token, _token_id in ordered_candidates if token in excluded]
    exact_collisions = [
        token
        for token, _token_id in ordered_candidates
        if token not in excluded and token in base_vocab
    ]
    selected_tokens = [
        token
        for token, _token_id in ordered_candidates
        if token and token not in excluded and token not in base_vocab
    ]
    if len(selected_tokens) != len(set(selected_tokens)):
        raise ValueError("Selected extension tokens must be unique.")
    return {
        "candidate_vocab_size": len(candidate_vocab),
        "excluded_special_tokens": special_tokens,
        "exact_collision_tokens": exact_collisions,
        "selected_tokens": selected_tokens,
    }


def verify_original_ids_stable(
    original_vocab: dict[str, int],
    extended_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Fail if any original token ID changed during extension or serialization."""
    extended_vocab = extended_tokenizer.get_vocab()
    changed = [
        token
        for token, token_id in original_vocab.items()
        if extended_vocab.get(token) != token_id
    ]
    if changed:
        preview = ", ".join(repr(token) for token in changed[:5])
        raise ValueError(f"Extension changed {len(changed)} original token IDs: {preview}")


def extension_resource_costs(
    *,
    base_vocab_size: int,
    added_tokens: int,
    replacement_vocab_size: int,
    hidden_size: int,
    untied_embeddings: bool,
    base_embedding_rows: int | None = None,
    pad_to_multiple_of: int = 1,
) -> dict[str, int | float | bool]:
    """Report incremental and total lexical-interface cost for one model shape."""
    matrices = 2 if untied_embeddings else 1
    extended_vocab_size = base_vocab_size + added_tokens
    base_rows = base_embedding_rows if base_embedding_rows is not None else base_vocab_size
    extended_rows = math.ceil(extended_vocab_size / pad_to_multiple_of) * pad_to_multiple_of
    replacement_rows = math.ceil(replacement_vocab_size / pad_to_multiple_of) * pad_to_multiple_of
    incremental_rows = max(extended_rows - base_rows, 0)
    incremental_parameters = matrices * incremental_rows * hidden_size
    total_extension_parameters = matrices * extended_rows * hidden_size
    replacement_parameters = matrices * replacement_rows * hidden_size
    return {
        "hidden_size": hidden_size,
        "untied_input_output_embeddings": untied_embeddings,
        "matrix_count": matrices,
        "base_vocab_size": base_vocab_size,
        "base_embedding_rows": base_rows,
        "added_tokens": added_tokens,
        "extended_vocab_size": extended_vocab_size,
        "extended_embedding_rows": extended_rows,
        "replacement_vocab_size": replacement_vocab_size,
        "replacement_embedding_rows": replacement_rows,
        "pad_to_multiple_of": pad_to_multiple_of,
        "incremental_embedding_rows": incremental_rows,
        "incremental_parameters": incremental_parameters,
        "incremental_fp16_mib": incremental_parameters * 2 / (1024**2),
        "incremental_fp32_mib": incremental_parameters * 4 / (1024**2),
        "total_extension_parameters": total_extension_parameters,
        "total_extension_fp16_mib": total_extension_parameters * 2 / (1024**2),
        "replacement_parameters": replacement_parameters,
        "replacement_fp16_mib": replacement_parameters * 2 / (1024**2),
    }


def _artifact_files(output_dir: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    ]


def build_extension_tokenizer(
    *,
    base_tokenizer: PreTrainedTokenizerBase,
    base_model_id: str,
    candidate_path: Path,
    output_dir: Path,
    hidden_size: int = 1024,
    untied_embeddings: bool = True,
    base_embedding_rows: int | None = None,
    base_model_tied_embeddings: bool | None = None,
) -> dict[str, object]:
    """Append the frozen novel Akan candidates and verify a lossless round trip."""
    candidate_tokenizer = load_candidate_tokenizer(candidate_path)
    selection = select_extension_tokens(
        candidate_tokenizer=candidate_tokenizer,
        base_tokenizer=base_tokenizer,
    )
    selected_tokens = list(selection["selected_tokens"])
    original_vocab = base_tokenizer.get_vocab().copy()
    original_length = len(base_tokenizer)
    if original_length != len(original_vocab):
        raise ValueError("Base tokenizer length and vocabulary mapping size must agree.")

    added_tokens = [
        AddedToken(
            token,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=False,
        )
        for token in selected_tokens
    ]
    added_count = base_tokenizer.add_tokens(added_tokens, special_tokens=False)
    if added_count != len(selected_tokens):
        raise ValueError(
            "Tokenizer did not append every selected token: "
            f"expected {len(selected_tokens)}, added {added_count}."
        )
    verify_original_ids_stable(original_vocab, base_tokenizer)
    for offset, token in enumerate(selected_tokens):
        expected_id = original_length + offset
        if base_tokenizer.convert_tokens_to_ids(token) != expected_id:
            raise ValueError(f"Extension token {token!r} was not appended at ID {expected_id}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    base_tokenizer.save_pretrained(output_dir, legacy_format=False)
    reloaded = AutoTokenizer.from_pretrained(output_dir, local_files_only=True, use_fast=True)
    verify_original_ids_stable(original_vocab, reloaded)
    if len(reloaded) != original_length + added_count:
        raise ValueError("Reloaded extension tokenizer has an unexpected vocabulary size.")
    for offset, token in enumerate(selected_tokens):
        if reloaded.convert_tokens_to_ids(token) != original_length + offset:
            raise ValueError(f"Reload changed the appended ID for {token!r}.")

    candidate_vocab_size = int(selection["candidate_vocab_size"])
    payload: dict[str, object] = {
        "schema_version": EXTENSION_SCHEMA_VERSION,
        "experiment_id": "qwen3_0.6b_vocab_extension_v32000",
        "strategy": "extension",
        "base_model_id": base_model_id,
        "base_tokenizer": {
            "name_or_path": base_tokenizer.name_or_path,
            "original_vocab_size": original_length,
            "original_vocab_sha256": _vocab_digest(original_vocab),
        },
        "candidate_tokenizer": {
            "path": candidate_path.as_posix(),
            "bytes": candidate_path.stat().st_size,
            "sha256": sha256_file(candidate_path),
            "vocab_size": candidate_vocab_size,
        },
        "selection": {
            "rule": (
                "Traverse the locked 32K mixed-BPE vocabulary in learned token-ID order; "
                "exclude the shared special tokens and exact base-vocabulary collisions; "
                "append every remaining token without normalization."
            ),
            "normalization": "disabled for added tokens; preserve exact candidate surfaces",
            "collision_policy": "exclude exact token-string matches in the base vocabulary",
            "special_token_policy": "preserve base special tokens; never append candidate specials",
            "candidate_vocab_size": candidate_vocab_size,
            "excluded_special_token_count": len(selection["excluded_special_tokens"]),
            "excluded_special_tokens": selection["excluded_special_tokens"],
            "exact_collision_count": len(selection["exact_collision_tokens"]),
            "exact_collision_tokens": selection["exact_collision_tokens"],
            "added_token_count": added_count,
            "selected_tokens": selected_tokens,
        },
        "output": {
            "directory": output_dir.as_posix(),
            "extended_vocab_size": len(reloaded),
            "original_ids_stable": True,
            "round_trip_verified": True,
            "files": _artifact_files(output_dir),
        },
        "model_interface": {
            "embedding_resize": "append rows; original rows and token IDs remain unchanged",
            "initialization": "mean of the original tokenizer subwords for each added surface",
            "output_head": (
                "initialize appended output rows with the same mean-subword rule; retain the "
                "controlled pipeline's untied input/output interface"
            ),
            "base_model_tied_embeddings": base_model_tied_embeddings,
            "controlled_pipeline_tied_embeddings": not untied_embeddings,
            "cost": extension_resource_costs(
                base_vocab_size=original_length,
                added_tokens=added_count,
                replacement_vocab_size=candidate_vocab_size,
                hidden_size=hidden_size,
                untied_embeddings=untied_embeddings,
                base_embedding_rows=base_embedding_rows,
                pad_to_multiple_of=64,
            ),
        },
    }
    return payload


def build_extension_from_pretrained(
    *,
    base_model_id: str,
    candidate_path: Path,
    output_dir: Path,
    local_files_only: bool,
    hidden_size: int | None = None,
    untied_embeddings: bool = True,
) -> dict[str, object]:
    """Load a base tokenizer and build its frozen Akan vocabulary extension."""
    base_config: Any = AutoConfig.from_pretrained(
        base_model_id,
        local_files_only=local_files_only,
    )
    base_tokenizer: Any = AutoTokenizer.from_pretrained(
        base_model_id,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if not base_tokenizer.is_fast:
        raise ValueError("Vocabulary extension requires a fast base tokenizer.")
    return build_extension_tokenizer(
        base_tokenizer=base_tokenizer,
        base_model_id=base_model_id,
        candidate_path=candidate_path,
        output_dir=output_dir,
        hidden_size=hidden_size or int(base_config.hidden_size),
        untied_embeddings=untied_embeddings,
        base_embedding_rows=int(base_config.vocab_size),
        base_model_tied_embeddings=bool(base_config.tie_word_embeddings),
    )


def build_intrinsic_extension_comparison(
    *,
    base_model_id: str,
    extension_dir: Path,
    replacement_path: Path,
    extension_metadata_path: Path,
    asr_test_path: Path,
    formal_test_path: Path,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 20260802,
    local_files_only: bool = True,
) -> dict[str, object]:
    """Compare original, extension, and replacement tokenizers on both frozen regimes."""
    original: Any = AutoTokenizer.from_pretrained(
        base_model_id,
        use_fast=True,
        local_files_only=local_files_only,
    )
    extension: Any = AutoTokenizer.from_pretrained(
        extension_dir,
        use_fast=True,
        local_files_only=True,
    )
    replacement = load_candidate_tokenizer(replacement_path)
    verify_original_ids_stable(original.get_vocab(), extension)

    metadata = json.loads(extension_metadata_path.read_text(encoding="utf-8"))
    datasets: dict[str, dict[str, Any]] = {
        "asr": {
            "path": asr_test_path,
            "texts": samples_to_texts(load_jsonl_samples(asr_test_path)),
        },
        "formal": {
            "path": formal_test_path,
            "texts": samples_to_texts(load_jsonl_samples(formal_test_path)),
        },
    }
    tokenizers = {
        "original": (original, base_model_id),
        "extension": (extension, extension_dir.as_posix()),
        "replacement": (replacement, replacement_path.as_posix()),
    }
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    paired: dict[str, dict[str, tuple[Any, Any]]] = {}
    for strategy, (tokenizer, reference) in tokenizers.items():
        metrics[strategy] = {}
        paired[strategy] = {}
        for regime, dataset in datasets.items():
            result, token_counts, word_counts = evaluate_tokenizer(
                tokenizer=tokenizer,
                tokenizer_name=strategy,
                tokenizer_reference=reference,
                texts=dataset["texts"],
                source_file=dataset["path"].as_posix(),
            )
            metrics[strategy][regime] = result
            paired[strategy][regime] = (token_counts, word_counts)

    comparisons: dict[str, dict[str, float]] = {}
    bootstrap: dict[str, dict[str, object]] = {
        "extension_minus_original": {},
        "replacement_minus_original": {},
        "extension_minus_replacement": {},
    }
    for regime in datasets:
        original_fertility = float(metrics["original"][regime]["fertility"])
        extension_fertility = float(metrics["extension"][regime]["fertility"])
        replacement_fertility = float(metrics["replacement"][regime]["fertility"])
        comparisons[regime] = {
            "original_fertility": original_fertility,
            "extension_fertility": extension_fertility,
            "replacement_fertility": replacement_fertility,
            "extension_reduction_vs_original_percent": (
                (original_fertility - extension_fertility) / original_fertility * 100
            ),
            "replacement_reduction_vs_original_percent": (
                (original_fertility - replacement_fertility) / original_fertility * 100
            ),
            "extension_minus_replacement_fertility": (
                extension_fertility - replacement_fertility
            ),
        }
        for label, candidate, baseline in (
            ("extension_minus_original", "extension", "original"),
            ("replacement_minus_original", "replacement", "original"),
            ("extension_minus_replacement", "extension", "replacement"),
        ):
            candidate_counts, word_counts = paired[candidate][regime]
            baseline_counts, baseline_words = paired[baseline][regime]
            if not (word_counts == baseline_words).all():
                raise RuntimeError("Intrinsic tokenizer comparisons must use paired words.")
            bootstrap[label][regime] = paired_bootstrap_fertility_difference(
                candidate_token_counts=candidate_counts,
                baseline_token_counts=baseline_counts,
                word_counts=word_counts,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )

    return {
        "schema_version": 1,
        "experiment_id": "qwen3_0.6b_extension_intrinsic_revision_v2",
        "question": (
            "How much intrinsic tokenization efficiency is retained when the locked 32K Akan "
            "candidate vocabulary extends, rather than replaces, the Qwen lexical interface?"
        ),
        "protocol": {
            "base_model_id": base_model_id,
            "extension_dir": extension_dir.as_posix(),
            "replacement_path": replacement_path.as_posix(),
            "asr_test": {
                "path": asr_test_path.as_posix(),
                "rows": len(datasets["asr"]["texts"]),
                "sha256": sha256_file(asr_test_path),
            },
            "formal_test": {
                "path": formal_test_path.as_posix(),
                "rows": len(datasets["formal"]["texts"]),
                "sha256": sha256_file(formal_test_path),
            },
            "bootstrap": {
                "method": "paired row resampling of aggregate fertility difference",
                "resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
                "confidence_level": 0.95,
            },
        },
        "extension_contract": {
            "selection": {
                key: metadata["selection"][key]
                for key in (
                    "rule",
                    "candidate_vocab_size",
                    "excluded_special_token_count",
                    "exact_collision_count",
                    "added_token_count",
                )
            },
            "output": metadata["output"],
            "model_interface": metadata["model_interface"],
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "bootstrap_confidence_intervals": bootstrap,
        "scope_decision": {
            "status": "intrinsic_complete_model_quality_pending",
            "model_quality_gate": (
                "Do not select extension or replacement from fertility alone; run the controlled "
                "Qwen BPB, chrF/chrF++, downstream, throughput, and checkpoint comparison."
            ),
        },
    }
