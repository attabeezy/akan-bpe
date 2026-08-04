"""Frozen run-matrix utilities for the revision-v2 GPU experiments."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from akan_bpe.model_integration import ModelIntegrationConfig, PeftConfigSpec


@dataclass(frozen=True)
class RevisionRunSpec:
    """One fully expanded GPU run."""

    run_id: str
    model_slug: str
    model_id: str
    strategy: str
    initialization: str
    seed: int
    vocab_size: int
    tokenizer_path: Path
    output_dir: Path
    result_path: Path
    goals: tuple[str, ...]


@dataclass(frozen=True)
class RevisionMatrix:
    """Validated matrix plus its expanded run list."""

    path: Path
    sha256: str
    payload: dict[str, Any]
    runs: tuple[RevisionRunSpec, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _epoch_tag(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _run_id(
    model_slug: str,
    strategy: str,
    vocab_size: int,
    initialization: str,
    epochs: float,
    seed: int,
) -> str:
    return (
        f"{model_slug}__{strategy}__v{vocab_size}__{initialization}"
        f"__e{_epoch_tag(epochs)}__s{seed}"
    )


def load_revision_matrix(path: Path) -> RevisionMatrix:
    """Load, validate, and deterministically expand a GPU matrix YAML file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Revision GPU matrix must be a YAML mapping.")
    payload = cast(dict[str, Any], raw)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported revision GPU matrix schema_version.")

    protocol = payload["protocol"]
    paths = payload["paths"]
    tokenizers = payload["tokenizers"]
    seeds = [int(seed) for seed in payload["seeds"]]
    if len(seeds) != len(set(seeds)) or len(seeds) < 3:
        raise ValueError("The revision matrix requires at least three unique seeds.")

    vocab_size = int(payload["vocab_size"])
    epochs = float(protocol["epochs"])
    runs: list[RevisionRunSpec] = []
    for arm in payload["arms"]:
        strategy = str(arm["strategy"])
        if strategy not in tokenizers:
            raise ValueError(f"No tokenizer path is configured for strategy {strategy!r}.")
        for seed in seeds:
            run_id = _run_id(
                model_slug=str(arm["model_slug"]),
                strategy=strategy,
                vocab_size=vocab_size,
                initialization=str(arm["initialization"]),
                epochs=epochs,
                seed=seed,
            )
            runs.append(
                RevisionRunSpec(
                    run_id=run_id,
                    model_slug=str(arm["model_slug"]),
                    model_id=str(arm["model_id"]),
                    strategy=strategy,
                    initialization=str(arm["initialization"]),
                    seed=seed,
                    vocab_size=vocab_size,
                    tokenizer_path=Path(tokenizers[strategy]),
                    output_dir=Path(paths["models_dir"]) / run_id,
                    result_path=Path(paths["results_dir"]) / f"{run_id}.json",
                    goals=tuple(str(goal) for goal in arm.get("goals", [])),
                )
            )

    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Expanded revision matrix contains duplicate run IDs.")
    expected_runs = int(payload["expected_runs"])
    if len(runs) != expected_runs:
        raise ValueError(f"Expected {expected_runs} runs but matrix expands to {len(runs)}.")
    return RevisionMatrix(path=path, sha256=_sha256(path), payload=payload, runs=tuple(runs))


def model_config_for_run(matrix: RevisionMatrix, run: RevisionRunSpec) -> ModelIntegrationConfig:
    """Translate one frozen matrix row into the existing single-run pipeline config."""
    protocol = matrix.payload["protocol"]
    generation = protocol["generation_evaluation"]
    qlora = protocol["qlora"]
    return ModelIntegrationConfig(
        experiment_id=run.run_id,
        model_id=run.model_id,
        tokenizer_path=run.tokenizer_path.as_posix(),
        tokenizer_strategy=run.strategy,
        train_file=str(protocol["train_file"]),
        eval_file=str(protocol["eval_file"]),
        output_dir=run.output_dir.as_posix(),
        results_output=run.result_path.as_posix(),
        device_mode=str(protocol["device_mode"]),
        max_train_samples=int(protocol["max_train_samples"]),
        max_eval_samples=int(protocol["max_eval_samples"]),
        max_length=int(protocol["max_length"]),
        batch_size=int(protocol["batch_size"]),
        grad_accum=int(protocol["gradient_accumulation_steps"]),
        epochs=float(protocol["epochs"]),
        learning_rate=float(protocol["learning_rate"]),
        optimizer=str(protocol["optimizer"]),
        lr_scheduler_type=str(protocol["lr_scheduler_type"]),
        warmup_ratio=float(protocol["warmup_ratio"]),
        weight_decay=float(protocol["weight_decay"]),
        max_grad_norm=float(protocol["max_grad_norm"]),
        peft=PeftConfigSpec(
            rank=int(qlora["rank"]),
            alpha=int(qlora["alpha"]),
            dropout=float(qlora["dropout"]),
            target_modules=tuple(str(module) for module in qlora["target_modules"]),
        ),
        seed=run.seed,
        generation_samples=int(protocol["generation_samples"]),
        generation_max_new_tokens=int(protocol["generation_max_new_tokens"]),
        generation_eval_samples=int(generation["examples"]),
        generation_prompt_words=int(generation["prompt_words"]),
        generation_reference_words=int(generation["reference_words"]),
        generation_eval_max_new_tokens=int(generation["max_new_tokens"]),
        generation_eval_batch_size=int(generation["batch_size"]),
        embedding_init_mode=run.initialization,
        compute_base_bpb=bool(protocol["compute_base_bpb"]),
    )


def _get_path(payload: dict[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


def validate_run_result(
    matrix: RevisionMatrix,
    run: RevisionRunSpec,
    payload: dict[str, Any],
    *,
    require_matrix_metadata: bool = True,
) -> None:
    """Reject result files that are incomplete or do not match their frozen run row."""
    config = model_config_for_run(matrix, run)
    expected = {
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
    }
    mismatches = [
        f"{key}: expected {value!r}, got {payload.get(key)!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    missing = []
    for dotted_path in matrix.payload["required_result_fields"]:
        try:
            value = _get_path(payload, str(dotted_path))
        except KeyError:
            missing.append(str(dotted_path))
            continue
        if value is None:
            missing.append(str(dotted_path))
    if require_matrix_metadata:
        metadata = payload.get("revision_matrix", {})
        if metadata.get("matrix_id") != matrix.payload["matrix_id"]:
            mismatches.append("revision_matrix.matrix_id does not match")
        if metadata.get("sha256") != matrix.sha256:
            mismatches.append("revision_matrix.sha256 does not match current YAML")
    if payload.get("reload_verification", {}).get("success") is not True:
        mismatches.append("reload_verification.success is not true")
    if mismatches or missing:
        details = mismatches + [f"missing required field: {field}" for field in missing]
        raise ValueError(f"Invalid result for {run.run_id}: " + "; ".join(details))


def result_status(matrix: RevisionMatrix) -> dict[str, list[str]]:
    """Classify expanded runs by artifact presence; validation is handled separately."""
    completed = [run.run_id for run in matrix.runs if run.result_path.exists()]
    pending = [run.run_id for run in matrix.runs if not run.result_path.exists()]
    return {"completed": completed, "pending": pending}


def _sample_summary(values_by_seed: dict[int, float]) -> dict[str, Any]:
    values = list(values_by_seed.values())
    return {
        "values_by_seed": {str(seed): value for seed, value in sorted(values_by_seed.items())},
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _paired_t_interval(deltas: list[float]) -> dict[str, float | int | str]:
    if len(deltas) != 3:
        raise ValueError("The frozen paired interval requires exactly three matched seeds.")
    mean = statistics.fmean(deltas)
    standard_deviation = statistics.stdev(deltas)
    margin = 4.302652729911275 * standard_deviation / math.sqrt(3)
    return {
        "method": "two_sided_paired_t_interval",
        "confidence_level": 0.95,
        "degrees_of_freedom": 2,
        "mean_delta": mean,
        "sample_standard_deviation": standard_deviation,
        "lower": mean - margin,
        "upper": mean + margin,
    }


def aggregate_matrix_results(matrix: RevisionMatrix) -> dict[str, Any]:
    """Validate all 15 artifacts and compute per-arm and paired-seed summaries."""
    loaded: dict[str, dict[str, Any]] = {}
    missing = [str(run.result_path) for run in matrix.runs if not run.result_path.exists()]
    if missing:
        raise ValueError(
            f"Cannot aggregate an incomplete matrix; missing {len(missing)} result(s)."
        )
    for run in matrix.runs:
        payload = cast(
            dict[str, Any], json.loads(run.result_path.read_text(encoding="utf-8"))
        )
        validate_run_result(matrix, run, payload)
        loaded[run.run_id] = payload

    metrics = matrix.payload["metrics"]
    arm_summaries: dict[str, Any] = {}
    for run in matrix.runs:
        arm_id = f"{run.model_slug}__{run.strategy}__{run.initialization}"
        arm = arm_summaries.setdefault(
            arm_id,
            {
                "model_slug": run.model_slug,
                "model_id": run.model_id,
                "strategy": run.strategy,
                "initialization": run.initialization,
                "metrics": {},
            },
        )
        for metric in metrics:
            name = str(metric["name"])
            arm["metrics"].setdefault(name, {})[run.seed] = float(
                _get_path(loaded[run.run_id], str(metric["path"]))
            )
    for arm in arm_summaries.values():
        arm["metrics"] = {
            name: _sample_summary(values) for name, values in arm["metrics"].items()
        }

    paired: dict[str, Any] = {}
    for comparison in matrix.payload["paired_comparisons"]:
        challenger_id = (
            f"{comparison['model_slug']}__{comparison['challenger']['strategy']}"
            f"__{comparison['challenger']['initialization']}"
        )
        reference_id = (
            f"{comparison['model_slug']}__{comparison['reference']['strategy']}"
            f"__{comparison['reference']['initialization']}"
        )
        result: dict[str, Any] = {
            "challenger": challenger_id,
            "reference": reference_id,
            "delta_definition": "challenger_minus_reference",
            "metrics": {},
        }
        for metric in metrics:
            name = str(metric["name"])
            challenger_values = arm_summaries[challenger_id]["metrics"][name]["values_by_seed"]
            reference_values = arm_summaries[reference_id]["metrics"][name]["values_by_seed"]
            seeds = sorted(set(challenger_values) & set(reference_values), key=int)
            deltas = [challenger_values[seed] - reference_values[seed] for seed in seeds]
            result["metrics"][name] = {
                "higher_is_better": bool(metric["higher_is_better"]),
                "deltas_by_seed": dict(zip(seeds, deltas)),
                "paired_confidence_interval": _paired_t_interval(deltas),
            }
        paired[str(comparison["id"])] = result

    return {
        "schema_version": 1,
        "matrix_id": matrix.payload["matrix_id"],
        "matrix_sha256": matrix.sha256,
        "status": "complete",
        "run_count": len(matrix.runs),
        "seeds": matrix.payload["seeds"],
        "arm_summaries": arm_summaries,
        "paired_comparisons": paired,
    }
