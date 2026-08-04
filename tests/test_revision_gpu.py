from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from akan_bpe.revision_gpu import (
    aggregate_matrix_results,
    load_revision_matrix,
    model_config_for_run,
    validate_run_result,
)


def test_frozen_matrix_expands_expected_15_unique_runs() -> None:
    matrix = load_revision_matrix(Path("config/revision_gpu_matrix.yaml"))

    assert len(matrix.runs) == 15
    assert len({run.run_id for run in matrix.runs}) == 15
    assert matrix.runs[0].run_id == "qwen-0.6b__replacement__v32000__random__e1__s17"
    extension = next(run for run in matrix.runs if run.strategy == "extension")
    config = model_config_for_run(matrix, extension)
    assert config.model_id == "Qwen/Qwen3-0.6B-Base"
    assert config.tokenizer_strategy == "extension"
    assert config.generation_eval_samples == 512
    assert config.seed == extension.seed


def _small_matrix(tmp_path: Path):
    payload = yaml.safe_load(Path("config/revision_gpu_matrix.yaml").read_text(encoding="utf-8"))
    payload["paths"] = {
        "results_dir": str(tmp_path / "results"),
        "models_dir": str(tmp_path / "models"),
        "aggregate_result": str(tmp_path / "aggregate.json"),
    }
    payload["arms"] = payload["arms"][:2]
    payload["expected_runs"] = 6
    payload["paired_comparisons"] = payload["paired_comparisons"][:1]
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_revision_matrix(matrix_path)


def _result_payload(matrix, run, *, bpb: float, chrf: float) -> dict[str, object]:
    config = model_config_for_run(matrix, run)
    return {
        "experiment_id": run.run_id,
        "model_id": run.model_id,
        "tokenizer_path": config.tokenizer_path,
        "tokenizer_strategy": run.strategy,
        "embedding_init_mode": run.initialization,
        "seed": run.seed,
        "train_file": config.train_file,
        "eval_file": config.eval_file,
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "lr_scheduler_type": config.lr_scheduler_type,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "eval": {
            "bpb": {
                "experiment": {"bits_per_byte": bpb},
                "base": {"bits_per_byte": 2.0},
            },
            "generation_quality": {"chrf": chrf, "chrfpp": chrf + 0.5},
        },
        "training": {
            "trainer_metrics": {"train_runtime": 100.0 + run.seed},
            "estimated_processed_non_padding_tokens": 1234,
            "checkpoint_bytes": 5000,
        },
        "reload_verification": {"success": True},
        "revision_matrix": {
            "matrix_id": matrix.payload["matrix_id"],
            "sha256": matrix.sha256,
        },
    }


def test_result_validation_rejects_wrong_seed(tmp_path: Path) -> None:
    matrix = _small_matrix(tmp_path)
    run = matrix.runs[0]
    payload = _result_payload(matrix, run, bpb=1.2, chrf=10.0)
    payload["seed"] = 999

    with pytest.raises(ValueError, match="seed"):
        validate_run_result(matrix, run, payload)


def test_aggregate_requires_completeness_and_reports_paired_intervals(tmp_path: Path) -> None:
    matrix = _small_matrix(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_matrix_results(matrix)

    for run in matrix.runs:
        is_mean = run.initialization == "mean_subword"
        payload = _result_payload(
            matrix,
            run,
            bpb=1.0 + run.seed / 1000 - (0.1 if is_mean else 0.0),
            chrf=10.0 + run.seed / 100 + (1.0 if is_mean else 0.0),
        )
        run.result_path.parent.mkdir(parents=True, exist_ok=True)
        run.result_path.write_text(json.dumps(payload), encoding="utf-8")

    aggregate = aggregate_matrix_results(matrix)
    comparison = aggregate["paired_comparisons"]["qwen-0.6b-mean-subword-minus-random"]
    bpb = comparison["metrics"]["full_coverage_bpb"]
    chrf = comparison["metrics"]["chrf"]

    assert aggregate["status"] == "complete"
    assert aggregate["run_count"] == 6
    assert bpb["paired_confidence_interval"]["mean_delta"] == pytest.approx(-0.1)
    assert chrf["paired_confidence_interval"]["mean_delta"] == pytest.approx(1.0)
    assert set(bpb["deltas_by_seed"]) == {"17", "42", "73"}
