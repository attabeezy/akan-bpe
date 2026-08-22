"""Frozen AfriSenti downstream-evaluation contracts and CPU-safe utilities."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class DownstreamRunSpec:
    """One base or recreated-adapter downstream evaluation."""

    run_id: str
    model_slug: str
    model_id: str
    strategy: str
    initialization: str | None
    seed: int | None
    source_run_id: str | None
    result_path: Path


@dataclass(frozen=True)
class DownstreamManifest:
    """Validated manifest and its deterministic expanded runs."""

    path: Path
    sha256: str
    payload: dict[str, Any]
    runs: tuple[DownstreamRunSpec, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(text: str) -> str:
    """Apply the frozen AfriSenti duplicate-audit normalization."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalized_text(text).encode("utf-8")).hexdigest()


def _source_run_id(model_slug: str, strategy: str, initialization: str, seed: int) -> str:
    return f"{model_slug}__{strategy}__v32000__{initialization}__e1__s{seed}"


def load_downstream_manifest(path: Path) -> DownstreamManifest:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Unsupported downstream manifest schema.")
    payload = cast(dict[str, Any], raw)
    labels = list(payload["dataset"]["labels"])
    if labels != ["negative", "neutral", "positive"]:
        raise ValueError("AfriSenti label order must be negative, neutral, positive.")
    if set(payload["prompt"]["label_codes"]) != set(labels):
        raise ValueError("Every dataset label requires one prompt code.")
    if len(payload["prompt"]["cyclic_orders"]) != 3:
        raise ValueError("Exactly three cyclic prompt orders are required.")

    result_dir = Path(payload["paths"]["results_dir"])
    runs: list[DownstreamRunSpec] = []
    for arm in payload["base_arms"]:
        slug = str(arm["model_slug"])
        run_id = f"afrisenti-twi__{slug}__original__base"
        runs.append(
            DownstreamRunSpec(
                run_id,
                slug,
                str(arm["model_id"]),
                "original",
                None,
                None,
                None,
                result_dir / f"{run_id}.json",
            )
        )
    matrix_model_ids = {
        "qwen-0.6b": "Qwen/Qwen3-0.6B-Base",
        "qwen-1.7b": "Qwen/Qwen3-1.7B-Base",
    }
    for arm in payload["adapted_arms"]:
        slug = str(arm["model_slug"])
        strategy = str(arm["strategy"])
        initialization = str(arm["initialization"])
        for raw_seed in payload["seeds"]:
            seed = int(raw_seed)
            source = _source_run_id(slug, strategy, initialization, seed)
            run_id = f"afrisenti-twi__{slug}__{strategy}__{initialization}__s{seed}"
            runs.append(
                DownstreamRunSpec(
                    run_id,
                    slug,
                    matrix_model_ids[slug],
                    strategy,
                    initialization,
                    seed,
                    source,
                    result_dir / f"{run_id}.json",
                )
            )
    if len(runs) != int(payload["expected_runs"]):
        raise ValueError("Downstream manifest does not expand to expected_runs.")
    ids = [run.run_id for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Downstream manifest expands duplicate run IDs.")
    return DownstreamManifest(path, sha256_file(path), payload, tuple(runs))


def audit_dataset_splits(
    splits: dict[str, list[dict[str, str]]], manifest: DownstreamManifest
) -> dict[str, Any]:
    """Validate schema/counts and return duplicate and clean-test membership."""
    dataset = manifest.payload["dataset"]
    labels = set(dataset["labels"])
    normalized_sets: dict[str, set[str]] = {}
    labels_by_hash: dict[str, dict[str, set[str]]] = {}
    counts: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        rows = splits.get(split, [])
        expected = int(dataset["splits"][split]["rows"])
        if len(rows) != expected:
            raise ValueError(f"{split} row count mismatch: expected {expected}, got {len(rows)}")
        invalid = [index for index, row in enumerate(rows) if row.get("label") not in labels]
        if invalid:
            raise ValueError(f"{split} contains invalid labels at rows {invalid[:5]}")
        hashes = [normalized_sha256(row["tweet"]) for row in rows]
        normalized_sets[split] = set(hashes)
        label_map: dict[str, set[str]] = {}
        for digest, row in zip(hashes, rows):
            label_map.setdefault(digest, set()).add(row["label"])
        labels_by_hash[split] = label_map
        counts[split] = {"rows": len(rows), "unique_normalized_texts": len(set(hashes))}
    train_validation = normalized_sets["train"] | normalized_sets["validation"]
    excluded = normalized_sets["test"] & train_validation
    clean_indices = [
        index
        for index, row in enumerate(splits["test"])
        if normalized_sha256(row["tweet"]) not in excluded
    ]
    if len(clean_indices) != int(dataset["clean_test_rows"]):
        raise ValueError(
            f"Clean test mismatch: expected {dataset['clean_test_rows']}, got {len(clean_indices)}"
        )
    overlaps = {}
    overlap_conflicts = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        key = f"{left}-{right}"
        shared = normalized_sets[left] & normalized_sets[right]
        overlaps[key] = len(shared)
        overlap_conflicts[key] = sum(
            labels_by_hash[left][digest] != labels_by_hash[right][digest] for digest in shared
        )
    within_conflicts = {
        split: sum(len(values) > 1 for values in labels_by_hash[split].values())
        for split in labels_by_hash
    }
    audit = {
        "counts": counts,
        "overlaps": overlaps,
        "overlap_label_conflicts": overlap_conflicts,
        "within_split_label_conflicts": within_conflicts,
        "clean_test_indices": clean_indices,
    }
    frozen = dataset.get("frozen_audit")
    if frozen:
        calculated = {
            "unique_normalized_texts": {
                split: counts[split]["unique_normalized_texts"] for split in counts
            },
            "overlaps": overlaps,
            "overlap_label_conflicts": overlap_conflicts,
            "within_split_label_conflicts": within_conflicts,
        }
        for key, value in calculated.items():
            if frozen.get(key) != value:
                raise ValueError(f"Frozen dataset audit mismatch for {key}.")
    return audit


def audit_adaptation_overlap(
    test_rows: list[dict[str, str]], paths: Iterable[Path]
) -> dict[str, int | None]:
    """Count normalized AfriSenti test matches in available adaptation corpora."""
    test_hashes = {normalized_sha256(row["tweet"]) for row in test_rows}
    result: dict[str, int | None] = {}
    for path in paths:
        if not path.exists():
            result[path.as_posix()] = None
            continue
        overlaps = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if normalized_sha256(str(row["text"])) in test_hashes:
                    overlaps += 1
        result[path.as_posix()] = overlaps
    return result


def fetch_dataset(manifest: DownstreamManifest) -> dict[str, list[dict[str, str]]]:
    """Download the three pinned parquet files, verify bytes, and return rows."""
    from datasets import Dataset

    dataset = manifest.payload["dataset"]
    cache = Path(manifest.payload["paths"]["dataset_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "validation", "test"):
        path = cache / f"{split}.parquet"
        if not path.exists() or sha256_file(path) != dataset["splits"][split]["sha256"]:
            url = (
                "https://huggingface.co/api/datasets/"
                f"{dataset['repository']}/parquet/{dataset['config']}/{split}/0.parquet"
            )
            with urllib.request.urlopen(url) as response:
                path.write_bytes(response.read())
        expected_sha = str(dataset["splits"][split]["sha256"])
        if sha256_file(path) != expected_sha:
            raise ValueError(f"Downloaded {split} parquet does not match its frozen SHA-256.")
        rows = Dataset.from_parquet(str(path)).to_list()
        splits[split] = [
            {
                "tweet": str(row[dataset["fields"]["text"]]),
                "label": str(row[dataset["fields"]["label"]]),
            }
            for row in rows
        ]
    audit_dataset_splits(splits, manifest)
    return splits


def resolve_demonstrations(
    train_rows: list[dict[str, str]], manifest: DownstreamManifest
) -> dict[str, dict[str, str]]:
    """Resolve and hash-check the three frozen demonstration rows."""
    resolved: dict[str, dict[str, str]] = {}
    for spec in manifest.payload["prompt"]["demonstrations"]:
        index = int(spec["row_index"])
        row = train_rows[index]
        label = str(spec["label"])
        if row["label"] != label:
            raise ValueError(f"Demonstration row {index} has unexpected label.")
        digest = normalized_sha256(row["tweet"])
        if digest != spec["normalized_sha256"]:
            raise ValueError(f"Demonstration row {index} has unexpected text hash.")
        resolved[label] = row
    return resolved


def validate_demonstration_selection(
    splits: dict[str, list[dict[str, str]]], manifest: DownstreamManifest
) -> None:
    """Recompute the seeded rank and verify the frozen demonstrations are its winners."""
    prompt = manifest.payload["prompt"]
    seed = int(prompt["selection_seed"])
    held_out = {
        normalized_sha256(row["tweet"]) for split in ("validation", "test") for row in splits[split]
    }
    frozen = {str(spec["label"]): spec for spec in prompt["demonstrations"]}
    for label in manifest.payload["dataset"]["labels"]:
        candidates = []
        for index, row in enumerate(splits["train"]):
            digest = normalized_sha256(row["tweet"])
            if row["label"] != label or not normalized_text(row["tweet"]) or digest in held_out:
                continue
            rank = hashlib.sha256(f"{seed}\0{label}\0{row['tweet']}".encode()).hexdigest()
            candidates.append((rank, index, digest))
        if not candidates:
            raise ValueError(f"No eligible demonstration for label {label}.")
        _rank, index, digest = min(candidates)
        spec = frozen[label]
        if int(spec["row_index"]) != index or spec["normalized_sha256"] != digest:
            raise ValueError(f"Frozen demonstration selection mismatch for {label}.")


def build_prompt(
    tweet: str,
    demonstrations: dict[str, dict[str, str]],
    order: Iterable[str],
    label_codes: dict[str, str],
) -> str:
    """Build the fixed language-minimal three-shot classification prompt."""
    lines = ["Twi sentiment labels: 0=negative, 1=neutral, 2=positive."]
    for label in order:
        lines.extend(
            [f"Tweet: {demonstrations[label]['tweet']}", f"Sentiment: {label_codes[label]}"]
        )
    lines.extend([f"Tweet: {tweet}", "Sentiment:"])
    return "\n".join(lines)


def validate_prompt_tokenizer(tokenizer: Any, manifest: DownstreamManifest) -> None:
    """Require equal-length candidate encodings for the active tokenizer."""
    prompt = manifest.payload["prompt"]
    candidates = [prompt["candidate_prefix"] + code for code in prompt["label_codes"].values()]
    lengths = [
        len(tokenizer.encode(candidate, add_special_tokens=False)) for candidate in candidates
    ]
    if not lengths or min(lengths) == 0 or len(set(lengths)) != 1:
        raise ValueError(f"Candidate labels must have equal non-zero token lengths; got {lengths}.")


def _candidate_scores(
    model: Any, tokenizer: Any, prompts: list[str], candidate: str, torch: Any
) -> list[float]:
    prompt_ids = [tokenizer.encode(text, add_special_tokens=True) for text in prompts]
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    sequences = [ids + candidate_ids for ids in prompt_ids]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token for batched scoring.")
    width = max(len(ids) for ids in sequences)
    input_ids = torch.tensor(
        [ids + [pad_id] * (width - len(ids)) for ids in sequences], dtype=torch.long
    ).to(model.device)
    attention = torch.tensor(
        [[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences], dtype=torch.long
    ).to(model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)
    scores = []
    for row, ids in enumerate(prompt_ids):
        score = 0.0
        for offset, token_id in enumerate(candidate_ids):
            score += float(log_probs[row, len(ids) + offset - 1, token_id].item())
        scores.append(score)
    return scores


def evaluate_model(
    model: Any,
    tokenizer: Any,
    test_rows: list[dict[str, str]],
    demonstrations: dict[str, dict[str, str]],
    manifest: DownstreamManifest,
) -> list[dict[str, Any]]:
    """Run cyclic-order candidate scoring and a greedy parse diagnostic."""
    import torch

    validate_prompt_tokenizer(tokenizer, manifest)
    prompt_config = manifest.payload["prompt"]
    label_codes = cast(dict[str, str], prompt_config["label_codes"])
    code_labels = {value: key for key, value in label_codes.items()}
    prefix = str(prompt_config["candidate_prefix"])
    max_tokens = int(prompt_config["max_prompt_tokens"])
    batch_size = int(manifest.payload["evaluation"]["batch_size"])
    totals = [{label: 0.0 for label in label_codes} for _ in test_rows]
    canonical_prompts: list[str] = []
    for order_index, order in enumerate(prompt_config["cyclic_orders"]):
        prompts = [
            build_prompt(row["tweet"], demonstrations, order, label_codes) for row in test_rows
        ]
        if order_index == 0:
            canonical_prompts = prompts
        lengths = [len(tokenizer.encode(text, add_special_tokens=True)) for text in prompts]
        oversized = [index for index, length in enumerate(lengths) if length > max_tokens]
        if oversized:
            raise ValueError(
                f"Prompt coverage exceeds {max_tokens} tokens at rows {oversized[:5]}."
            )
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            for label, code in label_codes.items():
                candidate_scores = _candidate_scores(model, tokenizer, batch, prefix + code, torch)
                for offset, score in enumerate(candidate_scores):
                    totals[start + offset][label] += score

    greedy_codes: list[str | None] = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for start in range(0, len(canonical_prompts), batch_size):
            batch = canonical_prompts[start : start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=int(prompt_config["greedy_max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            suffix = generated[:, encoded["input_ids"].shape[1] :]
            for row in suffix:
                text = tokenizer.decode(row, skip_special_tokens=True).strip()
                greedy_codes.append(text[0] if text and text[0] in code_labels else None)
    finally:
        tokenizer.padding_side = original_padding_side

    predictions = []
    rotations = len(prompt_config["cyclic_orders"])
    for index, (row, total_scores, greedy_code) in enumerate(zip(test_rows, totals, greedy_codes)):
        averaged = {label: score / rotations for label, score in total_scores.items()}
        predicted = max(label_codes, key=lambda label: averaged[label])
        predictions.append(
            {
                "row_index": index,
                "normalized_sha256": normalized_sha256(row["tweet"]),
                "gold_label": row["label"],
                "predicted_label": predicted,
                "candidate_log_likelihood": averaged,
                "greedy_label": code_labels.get(greedy_code) if greedy_code else None,
                "greedy_valid": greedy_code is not None,
            }
        )
    return predictions


def classification_metrics(
    gold: list[str], predicted: list[str], labels: list[str]
) -> dict[str, Any]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("Gold and prediction lists must be non-empty and equal length.")
    confusion = [[0 for _ in labels] for _ in labels]
    label_index = {label: index for index, label in enumerate(labels)}
    for truth, guess in zip(gold, predicted):
        confusion[label_index[truth]][label_index[guess]] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, label in enumerate(labels):
        tp = confusion[index][index]
        fp = sum(row[index] for row in confusion) - tp
        fn = sum(confusion[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(confusion[index]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "accuracy": sum(confusion[i][i] for i in range(len(labels))) / len(gold),
        "macro_f1": statistics.fmean(f1_values),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "rows_true_columns_predicted": confusion},
        "examples": len(gold),
    }


def stratified_bootstrap_interval(
    gold: list[str],
    predicted: list[str],
    labels: list[str],
    metric: str,
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Compute a deterministic class-stratified percentile interval."""
    rng = random.Random(seed)
    by_label = {label: [i for i, value in enumerate(gold) if value == label] for label in labels}
    values = []
    for _ in range(resamples):
        sample = [rng.choice(indices) for indices in by_label.values() for _ in indices]
        metrics = classification_metrics(
            [gold[i] for i in sample], [predicted[i] for i in sample], labels
        )
        values.append(float(metrics[metric]))
    values.sort()
    lower_index = max(0, math.floor(0.025 * (len(values) - 1)))
    upper_index = min(len(values) - 1, math.ceil(0.975 * (len(values) - 1)))
    return {
        "method": "class_stratified_percentile_bootstrap",
        "confidence_level": 0.95,
        "resamples": resamples,
        "seed": seed,
        "lower": values[lower_index],
        "upper": values[upper_index],
    }


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    manifest: DownstreamManifest,
    clean_test_indices: Iterable[int],
) -> dict[str, Any]:
    labels = list(manifest.payload["dataset"]["labels"])
    clean = set(clean_test_indices)
    settings = manifest.payload["evaluation"]
    surfaces: dict[str, Any] = {}
    for surface in settings["surfaces"]:
        rows = (
            predictions
            if surface == "official"
            else [p for p in predictions if p["row_index"] in clean]
        )
        gold = [str(row["gold_label"]) for row in rows]
        predicted = [str(row["predicted_label"]) for row in rows]
        metrics = classification_metrics(gold, predicted, labels)
        metrics["macro_f1_interval"] = stratified_bootstrap_interval(
            gold,
            predicted,
            labels,
            "macro_f1",
            resamples=int(settings["bootstrap_resamples"]),
            seed=int(settings["bootstrap_seed"]),
        )
        metrics["accuracy_interval"] = stratified_bootstrap_interval(
            gold,
            predicted,
            labels,
            "accuracy",
            resamples=int(settings["bootstrap_resamples"]),
            seed=int(settings["bootstrap_seed"]),
        )
        metrics["invalid_output_rate"] = sum(
            not bool(row.get("greedy_valid")) for row in rows
        ) / len(rows)
        surfaces[str(surface)] = metrics
    return surfaces


def validate_result(
    manifest: DownstreamManifest, run: DownstreamRunSpec, payload: dict[str, Any]
) -> None:
    expected = {
        "run_id": run.run_id,
        "model_id": run.model_id,
        "strategy": run.strategy,
        "initialization": run.initialization,
        "seed": run.seed,
        "source_run_id": run.source_run_id,
        "manifest_sha256": manifest.sha256,
        "dataset_revision": manifest.payload["dataset"]["revision"],
    }
    errors = [f"{key} mismatch" for key, value in expected.items() if payload.get(key) != value]
    predictions = payload.get("predictions")
    expected_rows = int(manifest.payload["dataset"]["splits"]["test"]["rows"])
    labels = set(manifest.payload["dataset"]["labels"])
    if not isinstance(predictions, list) or len(predictions) != expected_rows:
        errors.append("predictions must cover the complete official test split")
    else:
        indices = {row.get("row_index") for row in predictions}
        if indices != set(range(expected_rows)):
            errors.append("prediction row indices must cover 0..test_rows-1 exactly")
        invalid_labels = [
            row.get("row_index")
            for row in predictions
            if row.get("gold_label") not in labels or row.get("predicted_label") not in labels
        ]
        if invalid_labels:
            errors.append(f"invalid prediction labels at rows {invalid_labels[:5]}")
    for surface in ("official", "clean"):
        metrics = payload.get("metrics", {}).get(surface)
        if not isinstance(metrics, dict):
            errors.append(f"missing metrics surface {surface}")
            continue
        expected_surface_rows = (
            expected_rows
            if surface == "official"
            else int(manifest.payload["dataset"]["clean_test_rows"])
        )
        if metrics.get("examples") != expected_surface_rows:
            errors.append(f"{surface} metrics coverage mismatch")
    if errors:
        raise ValueError(f"Invalid downstream result for {run.run_id}: " + "; ".join(errors))


def _paired_t_interval(values: list[float]) -> dict[str, float | int | str]:
    if len(values) != 3:
        raise ValueError("Paired downstream intervals require exactly three seeds.")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    margin = 4.302652729911275 * sd / math.sqrt(3)
    return {
        "method": "paired_t_interval",
        "confidence_level": 0.95,
        "degrees_of_freedom": 2,
        "mean_delta": mean,
        "sample_standard_deviation": sd,
        "lower": mean - margin,
        "upper": mean + margin,
    }


def aggregate_results(manifest: DownstreamManifest) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    for run in manifest.runs:
        if not run.result_path.exists():
            raise ValueError(f"Missing downstream result: {run.result_path}")
        payload = json.loads(run.result_path.read_text(encoding="utf-8"))
        validate_result(manifest, run, payload)
        loaded[run.run_id] = payload
    summaries: dict[str, Any] = {}
    for run in manifest.runs:
        arm = f"{run.model_slug}__{run.strategy}"
        values = summaries.setdefault(arm, {"runs": {}, "model_id": run.model_id})["runs"]
        values["base" if run.seed is None else str(run.seed)] = {
            surface: {
                metric: loaded[run.run_id]["metrics"][surface][metric]
                for metric in ("macro_f1", "accuracy")
            }
            for surface in ("official", "clean")
        }
    for summary in summaries.values():
        seed_runs = [result for seed, result in summary["runs"].items() if seed != "base"]
        if seed_runs:
            summary["seed_summary"] = {
                surface: {
                    metric: {
                        "mean": statistics.fmean(result[surface][metric] for result in seed_runs),
                        "sample_standard_deviation": statistics.stdev(
                            result[surface][metric] for result in seed_runs
                        ),
                        "n": len(seed_runs),
                    }
                    for metric in ("macro_f1", "accuracy")
                }
                for surface in ("official", "clean")
            }
    comparisons = {}
    pairs = {
        "qwen-0.6b-extension-minus-replacement": ("qwen-0.6b__extension", "qwen-0.6b__replacement"),
        "qwen-1.7b-minus-qwen-0.6b-replacement": (
            "qwen-1.7b__replacement",
            "qwen-0.6b__replacement",
        ),
        "qwen-0.6b-replacement-minus-base": (
            "qwen-0.6b__replacement",
            "qwen-0.6b__original",
        ),
        "qwen-0.6b-extension-minus-base": (
            "qwen-0.6b__extension",
            "qwen-0.6b__original",
        ),
        "qwen-1.7b-replacement-minus-base": (
            "qwen-1.7b__replacement",
            "qwen-1.7b__original",
        ),
    }
    for name, (challenger, reference) in pairs.items():
        result: dict[str, Any] = {}
        for surface in ("official", "clean"):
            result[surface] = {}
            for metric in ("macro_f1", "accuracy"):
                deltas = []
                for seed in manifest.payload["seeds"]:
                    reference_runs = summaries[reference]["runs"]
                    reference_result = reference_runs.get(str(seed)) or reference_runs["base"]
                    deltas.append(
                        summaries[challenger]["runs"][str(seed)][surface][metric]
                        - reference_result[surface][metric]
                    )
                result[surface][metric] = {
                    "deltas_by_seed": dict(zip(map(str, manifest.payload["seeds"]), deltas)),
                    "paired_interval": _paired_t_interval(deltas),
                }
        comparisons[name] = result
    return {
        "schema_version": 1,
        "experiment_id": manifest.payload["experiment_id"],
        "manifest_sha256": manifest.sha256,
        "status": "complete",
        "run_count": len(loaded),
        "arm_summaries": summaries,
        "paired_comparisons": comparisons,
    }


def render_markdown_table(aggregate: dict[str, Any]) -> str:
    lines = [
        "# AfriSenti Twi downstream results",
        "",
        "| Arm | Seed | Official macro-F1 | Official accuracy | Clean macro-F1 | Clean accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, summary in aggregate["arm_summaries"].items():
        for seed, metrics in summary["runs"].items():
            official = metrics["official"]
            clean = metrics["clean"]
            lines.append(
                f"| {arm} | {seed} | {official['macro_f1']:.4f} | "
                f"{official['accuracy']:.4f} | {clean['macro_f1']:.4f} | "
                f"{clean['accuracy']:.4f} |"
            )
        if "seed_summary" in summary:
            official = summary["seed_summary"]["official"]
            clean = summary["seed_summary"]["clean"]
            lines.append(
                f"| {arm} | mean | {official['macro_f1']['mean']:.4f} | "
                f"{official['accuracy']['mean']:.4f} | "
                f"{clean['macro_f1']['mean']:.4f} | {clean['accuracy']['mean']:.4f} |"
            )
    return "\n".join(lines) + "\n"
