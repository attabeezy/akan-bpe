from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from akan_bpe.model_integration import derive_experiment_id, load_experiment_tokenizer
from akan_bpe.tokenizers import train_bpe_tokenizer
from akan_bpe.vocab_extension import (
    build_extension_tokenizer,
    build_intrinsic_extension_comparison,
    extension_resource_costs,
    load_candidate_tokenizer,
    select_extension_tokens,
)


def _base_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            vocab={"<pad>": 0, "<unk>": 1, "shared": 2, "base": 3},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
    )


def _candidate_path(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.json"
    train_bpe_tokenizer(
        texts=[
            "shared akan extension token",
            "akan kasa token foforo",
            "extension kasa shared",
        ],
        output_path=path,
        vocab_size=64,
        name="candidate",
    )
    return path


def test_selection_excludes_specials_and_exact_base_collisions(tmp_path: Path) -> None:
    candidate = load_candidate_tokenizer(_candidate_path(tmp_path))
    base = _base_tokenizer()

    result = select_extension_tokens(
        candidate_tokenizer=candidate,
        base_tokenizer=base,
    )

    selected = result["selected_tokens"]
    assert "shared" not in selected
    assert "shared" in result["exact_collision_tokens"]
    assert not set(candidate.all_special_tokens).intersection(selected)
    candidate_vocab = candidate.get_vocab()
    assert selected == sorted(selected, key=candidate_vocab.__getitem__)


def test_extension_preserves_original_ids_and_round_trips(tmp_path: Path) -> None:
    candidate_path = _candidate_path(tmp_path)
    output_dir = tmp_path / "extended"
    base = _base_tokenizer()
    original_vocab = base.get_vocab().copy()

    payload = build_extension_tokenizer(
        base_tokenizer=base,
        base_model_id="fixture/base",
        candidate_path=candidate_path,
        output_dir=output_dir,
        hidden_size=8,
        untied_embeddings=True,
    )
    reloaded = AutoTokenizer.from_pretrained(output_dir, local_files_only=True, use_fast=True)
    integration_loaded = load_experiment_tokenizer(output_dir)

    assert payload["strategy"] == "extension"
    assert payload["output"]["original_ids_stable"] is True  # type: ignore[index]
    assert payload["output"]["round_trip_verified"] is True  # type: ignore[index]
    assert all(
        reloaded.get_vocab()[token] == token_id for token, token_id in original_vocab.items()
    )
    assert (
        len(reloaded)
        == len(original_vocab) + payload["selection"]["added_token_count"]  # type: ignore[index]
    )
    assert len(integration_loaded) == len(reloaded)


def test_extension_experiment_id_uses_strategy_tag() -> None:
    assert (
        derive_experiment_id("Qwen/Qwen3-0.6B-Base", "mean_subword", "extension")
        == "run-qwen-0.6b-extension-meansub"
    )


def test_resource_cost_reports_incremental_and_total_interfaces() -> None:
    result = extension_resource_costs(
        base_vocab_size=100,
        added_tokens=20,
        replacement_vocab_size=32,
        hidden_size=8,
        untied_embeddings=True,
    )

    assert result["incremental_parameters"] == 320
    assert result["total_extension_parameters"] == 1920
    assert result["replacement_parameters"] == 512

    padded = extension_resource_costs(
        base_vocab_size=100,
        base_embedding_rows=128,
        added_tokens=20,
        replacement_vocab_size=32,
        hidden_size=8,
        untied_embeddings=True,
        pad_to_multiple_of=64,
    )
    assert padded["extended_embedding_rows"] == 128
    assert padded["incremental_embedding_rows"] == 0


def test_intrinsic_comparison_uses_paired_frozen_regimes(tmp_path: Path) -> None:
    candidate_path = _candidate_path(tmp_path)
    base_dir = tmp_path / "base"
    extension_dir = tmp_path / "extended"
    metadata_path = tmp_path / "metadata.json"
    base = _base_tokenizer()
    base.save_pretrained(base_dir, legacy_format=False)
    metadata = build_extension_tokenizer(
        base_tokenizer=base,
        base_model_id=base_dir.as_posix(),
        candidate_path=candidate_path,
        output_dir=extension_dir,
        hidden_size=8,
        untied_embeddings=True,
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    asr_path = tmp_path / "asr.jsonl"
    formal_path = tmp_path / "formal.jsonl"
    asr_path.write_text(
        json.dumps({"id": "a", "text": "akan shared token", "source": "asr"}) + "\n",
        encoding="utf-8",
    )
    formal_path.write_text(
        json.dumps({"id": "f", "text": "kasa extension foforo", "source": "formal"}) + "\n",
        encoding="utf-8",
    )

    payload = build_intrinsic_extension_comparison(
        base_model_id=base_dir.as_posix(),
        extension_dir=extension_dir,
        replacement_path=candidate_path,
        extension_metadata_path=metadata_path,
        asr_test_path=asr_path,
        formal_test_path=formal_path,
        bootstrap_resamples=20,
        bootstrap_seed=7,
        local_files_only=True,
    )

    assert payload["metrics"]["extension"]["asr"]["num_samples"] == 1  # type: ignore[index]
    assert payload["metrics"]["replacement"]["formal"]["num_samples"] == 1  # type: ignore[index]
    assert payload["scope_decision"]["status"] == (  # type: ignore[index]
        "intrinsic_complete_model_quality_pending"
    )
