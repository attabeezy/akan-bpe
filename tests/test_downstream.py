from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from akan_bpe.downstream import (
    DownstreamManifest,
    aggregate_results,
    audit_dataset_splits,
    build_prompt,
    classification_metrics,
    load_downstream_manifest,
    normalized_sha256,
    resolve_demonstrations,
    stratified_bootstrap_interval,
    validate_demonstration_selection,
    validate_result,
)


def test_manifest_expands_two_bases_and_nine_adapted_runs() -> None:
    manifest = load_downstream_manifest(Path("config/downstream_afrisenti.yaml"))
    assert len(manifest.runs) == 11
    assert sum(run.seed is None for run in manifest.runs) == 2
    assert sum(run.seed is not None for run in manifest.runs) == 9
    assert manifest.runs[0].run_id == "afrisenti-twi__qwen-0.6b__original__base"
    assert manifest.runs[-1].source_run_id == (
        "qwen-1.7b__replacement__v32000__mean_subword__e1__s73"
    )


def _small_manifest() -> tuple[DownstreamManifest, dict[str, list[dict[str, str]]]]:
    manifest = load_downstream_manifest(Path("config/downstream_afrisenti.yaml"))
    payload = copy.deepcopy(manifest.payload)
    payload["dataset"].pop("frozen_audit", None)
    splits = {
        "train": [
            {"tweet": "bad", "label": "negative"},
            {"tweet": "plain", "label": "neutral"},
            {"tweet": "good", "label": "positive"},
            {"tweet": "awful", "label": "negative"},
        ],
        "validation": [{"tweet": "seen", "label": "neutral"}],
        "test": [
            {"tweet": "bad", "label": "negative"},
            {"tweet": "new", "label": "neutral"},
            {"tweet": "fresh", "label": "positive"},
        ],
    }
    for split, rows in splits.items():
        payload["dataset"]["splits"][split]["rows"] = len(rows)
    payload["dataset"]["clean_test_rows"] = 2
    payload["prompt"]["demonstrations"] = [
        {
            "label": splits["train"][index]["label"],
            "row_index": index,
            "normalized_sha256": normalized_sha256(splits["train"][index]["tweet"]),
        }
        for index in (3, 1, 2)
    ]
    return DownstreamManifest(manifest.path, manifest.sha256, payload, manifest.runs), splits


def test_dataset_audit_builds_clean_surface_and_resolves_demos() -> None:
    manifest, splits = _small_manifest()
    audit = audit_dataset_splits(splits, manifest)
    assert audit["overlaps"]["train-test"] == 1
    assert audit["clean_test_indices"] == [1, 2]
    demos = resolve_demonstrations(splits["train"], manifest)
    validate_demonstration_selection(splits, manifest)
    prompt = build_prompt(
        "fresh",
        demos,
        ["negative", "neutral", "positive"],
        manifest.payload["prompt"]["label_codes"],
    )
    assert prompt.endswith("Tweet: fresh\nSentiment:")
    assert "Tweet: awful\nSentiment: 0" in prompt


def test_classification_metrics_are_hand_checkable() -> None:
    labels = ["negative", "neutral", "positive"]
    metrics = classification_metrics(labels, ["negative", "positive", "positive"], labels)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["negative"]["f1"] == 1.0
    assert metrics["per_class"]["neutral"]["f1"] == 0.0
    assert metrics["per_class"]["positive"]["f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(5 / 9)


def test_stratified_bootstrap_is_reproducible() -> None:
    labels = ["negative", "neutral", "positive"]
    gold = labels * 3
    predicted = ["negative", "neutral", "negative"] * 3
    first = stratified_bootstrap_interval(
        gold, predicted, labels, "macro_f1", resamples=50, seed=17
    )
    second = stratified_bootstrap_interval(
        gold, predicted, labels, "macro_f1", resamples=50, seed=17
    )
    assert first == second


def _fake_result(manifest: DownstreamManifest, run, score: float) -> dict[str, object]:
    official = {"macro_f1": score, "accuracy": score, "examples": 949}
    clean = {"macro_f1": score, "accuracy": score, "examples": 730}
    return {
        "run_id": run.run_id,
        "model_id": run.model_id,
        "strategy": run.strategy,
        "initialization": run.initialization,
        "seed": run.seed,
        "source_run_id": run.source_run_id,
        "manifest_sha256": manifest.sha256,
        "dataset_revision": manifest.payload["dataset"]["revision"],
        "predictions": [
            {
                "row_index": index,
                "gold_label": "negative",
                "predicted_label": "negative",
            }
            for index in range(949)
        ],
        "metrics": {"official": official, "clean": clean},
    }


def test_validation_and_aggregation_require_complete_matching_artifacts(tmp_path: Path) -> None:
    original = load_downstream_manifest(Path("config/downstream_afrisenti.yaml"))
    runs = tuple(replace(run, result_path=tmp_path / f"{run.run_id}.json") for run in original.runs)
    manifest = DownstreamManifest(original.path, original.sha256, original.payload, runs)
    with pytest.raises(ValueError, match="Missing downstream result"):
        aggregate_results(manifest)
    for run in runs:
        score = 0.4 if run.seed is None else 0.5 + run.seed / 1000
        run.result_path.write_text(json.dumps(_fake_result(manifest, run, score)), encoding="utf-8")
    aggregate = aggregate_results(manifest)
    assert aggregate["run_count"] == 11
    assert "qwen-0.6b-extension-minus-replacement" in aggregate["paired_comparisons"]

    bad = _fake_result(manifest, runs[0], 0.4)
    bad["dataset_revision"] = "wrong"
    with pytest.raises(ValueError, match="dataset_revision mismatch"):
        validate_result(manifest, runs[0], bad)
