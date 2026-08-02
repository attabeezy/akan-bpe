"""Compare historical v1 and corrected v2 tokenizer/router evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASELINE_TOKENIZERS = (
    "xlm_roberta_base",
    "bert_base_multilingual_cased",
    "mt5_base",
)
AKAN_TOKENIZERS = ("asr", "tts", "mixed")
TEST_SETS = ("asr_test", "tts_test")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _fertility(payload: dict[str, Any], tokenizer: str, test_set: str) -> float:
    return float(payload["results"][tokenizer][test_set]["fertility"])


def _reduction_percent(custom_fertility: float, baseline_fertility: float) -> float:
    return (baseline_fertility - custom_fertility) / baseline_fertility * 100


def _router_summary(payload: dict[str, Any]) -> dict[str, int | float]:
    total = int(payload["total_samples"])
    correct = int(payload["routing_decisions"]["asr"])
    accuracy = correct / total * 100 if total else 0.0
    return {
        "samples": total,
        "correct": correct,
        "errors": total - correct,
        "accuracy_percent": accuracy,
    }


def _mixed_deployment_viable(payload: dict[str, Any]) -> bool:
    return all(
        _fertility(payload, "mixed", test_set)
        < min(_fertility(payload, baseline, test_set) for baseline in BASELINE_TOKENIZERS)
        for test_set in TEST_SETS
    )


def build_revision_comparison(
    *,
    fertility_v1_path: Path,
    fertility_v2_path: Path,
    heuristic_router_v1_path: Path,
    heuristic_router_v2_path: Path,
    ml_router_v1_path: Path,
    ml_router_v2_path: Path,
    max_fertility_delta: float = 0.01,
    max_reduction_delta_pp: float = 0.5,
    max_router_delta_pp: float = 0.5,
) -> dict[str, object]:
    """Build a decision-ready comparison from preserved v1 and v2 result artifacts."""
    fertility_v1 = _load_json(fertility_v1_path)
    fertility_v2 = _load_json(fertility_v2_path)
    if fertility_v1["test_sets"]["tts_test"] != fertility_v2["test_sets"]["tts_test"]:
        raise ValueError("v1 and v2 must use the same formal-Twi test file.")
    if fertility_v2["test_sets"]["asr_test"] == fertility_v1["test_sets"]["asr_test"]:
        raise ValueError("v2 must reference a distinct corrected ASR test file.")

    fertility_changes: dict[str, dict[str, object]] = {}
    absolute_fertility_deltas: list[float] = []
    for tokenizer in (*BASELINE_TOKENIZERS, *AKAN_TOKENIZERS):
        fertility_changes[tokenizer] = {}
        for test_set in TEST_SETS:
            v1_value = _fertility(fertility_v1, tokenizer, test_set)
            v2_value = _fertility(fertility_v2, tokenizer, test_set)
            delta = v2_value - v1_value
            absolute_fertility_deltas.append(abs(delta))
            fertility_changes[tokenizer][test_set] = {
                "v1": v1_value,
                "v2": v2_value,
                "absolute_delta": delta,
                "relative_delta_percent": delta / v1_value * 100 if v1_value else 0.0,
            }

    reduction_changes: dict[str, dict[str, dict[str, object]]] = {}
    absolute_reduction_deltas: list[float] = []
    for custom in AKAN_TOKENIZERS:
        reduction_changes[custom] = {}
        for baseline in BASELINE_TOKENIZERS:
            reduction_changes[custom][baseline] = {}
            for test_set in TEST_SETS:
                v1_reduction = _reduction_percent(
                    _fertility(fertility_v1, custom, test_set),
                    _fertility(fertility_v1, baseline, test_set),
                )
                v2_reduction = _reduction_percent(
                    _fertility(fertility_v2, custom, test_set),
                    _fertility(fertility_v2, baseline, test_set),
                )
                delta = v2_reduction - v1_reduction
                absolute_reduction_deltas.append(abs(delta))
                reduction_changes[custom][baseline][test_set] = {
                    "v1_percent": v1_reduction,
                    "v2_percent": v2_reduction,
                    "delta_percentage_points": delta,
                }

    routers: dict[str, dict[str, object]] = {}
    absolute_router_deltas: list[float] = []
    for name, v1_path, v2_path in (
        ("heuristic", heuristic_router_v1_path, heuristic_router_v2_path),
        ("ml", ml_router_v1_path, ml_router_v2_path),
    ):
        v1_summary = _router_summary(_load_json(v1_path))
        v2_summary = _router_summary(_load_json(v2_path))
        delta = float(v2_summary["accuracy_percent"]) - float(v1_summary["accuracy_percent"])
        absolute_router_deltas.append(abs(delta))
        routers[name] = {
            "v1": v1_summary,
            "v2": v2_summary,
            "accuracy_delta_percentage_points": delta,
        }

    best_v1 = fertility_v1["summary"]
    best_v2 = fertility_v2["summary"]
    best_unchanged = best_v1 == best_v2
    mixed_v1 = _mixed_deployment_viable(fertility_v1)
    mixed_v2 = _mixed_deployment_viable(fertility_v2)
    observed = {
        "max_absolute_fertility_delta": max(absolute_fertility_deltas, default=0.0),
        "max_absolute_reduction_delta_percentage_points": max(
            absolute_reduction_deltas, default=0.0
        ),
        "max_absolute_router_delta_percentage_points": max(absolute_router_deltas, default=0.0),
    }
    thresholds = {
        "max_fertility_delta": max_fertility_delta,
        "max_reduction_delta_percentage_points": max_reduction_delta_pp,
        "max_router_delta_percentage_points": max_router_delta_pp,
        "require_best_tokenizers_unchanged": True,
        "require_mixed_deployment_conclusion_unchanged": True,
    }
    material_change = (
        observed["max_absolute_fertility_delta"] > max_fertility_delta
        or observed["max_absolute_reduction_delta_percentage_points"] > max_reduction_delta_pp
        or observed["max_absolute_router_delta_percentage_points"] > max_router_delta_pp
        or not best_unchanged
        or mixed_v1 != mixed_v2
    )

    return {
        "schema_version": 1,
        "comparison_id": "asr_split_revision_v1_vs_v2",
        "inputs": {
            "fertility_v1": fertility_v1_path.as_posix(),
            "fertility_v2": fertility_v2_path.as_posix(),
            "heuristic_router_v1": heuristic_router_v1_path.as_posix(),
            "heuristic_router_v2": heuristic_router_v2_path.as_posix(),
            "ml_router_v1": ml_router_v1_path.as_posix(),
            "ml_router_v2": ml_router_v2_path.as_posix(),
        },
        "fertility_changes": fertility_changes,
        "reduction_changes": reduction_changes,
        "routers": routers,
        "decision": {
            "thresholds": thresholds,
            "observed": observed,
            "best_tokenizers": {"v1": best_v1, "v2": best_v2, "unchanged": best_unchanged},
            "mixed_deployment_viable": {
                "v1": mixed_v1,
                "v2": mixed_v2,
                "unchanged": mixed_v1 == mixed_v2,
            },
            "material_change": material_change,
            "model_ladder_disposition": (
                "retain_existing_model_ladder"
                if not material_change
                else "stop_and_investigate_before_new_experiments"
            ),
        },
        "router_limitation": (
            "The historical classifier was serialized under scikit-learn 1.8.0 and "
            "evaluated under 1.9.0 with an InconsistentVersionWarning."
        ),
    }


def render_revision_comparison_markdown(payload: dict[str, Any]) -> str:
    """Render the compact checkpoint tables directly from comparison JSON data."""
    lines = [
        "# ASR Split Revision: v1 vs v2",
        "",
        "## Fertility",
        "",
        "| Tokenizer | Test set | v1 | v2 | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for tokenizer, test_sets in payload["fertility_changes"].items():
        for test_set, metrics in test_sets.items():
            lines.append(
                f"| {tokenizer} | {test_set} | {metrics['v1']:.6f} | "
                f"{metrics['v2']:.6f} | {metrics['absolute_delta']:+.6f} |"
            )
    lines.extend(
        [
            "",
            "## Router",
            "",
            "| Router | v1 correct/total | v2 correct/total | v1 accuracy | v2 accuracy | Δ pp |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, comparison in payload["routers"].items():
        v1 = comparison["v1"]
        v2 = comparison["v2"]
        lines.append(
            f"| {name} | {v1['correct']}/{v1['samples']} | {v2['correct']}/{v2['samples']} | "
            f"{v1['accuracy_percent']:.4f}% | {v2['accuracy_percent']:.4f}% | "
            f"{comparison['accuracy_delta_percentage_points']:+.4f} |"
        )
    decision = payload["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Material change: **{str(decision['material_change']).lower()}**",
            "- Best tokenizers unchanged: "
            f"**{str(decision['best_tokenizers']['unchanged']).lower()}**",
            "- Mixed-tokenizer deployment conclusion unchanged: "
            f"**{str(decision['mixed_deployment_viable']['unchanged']).lower()}**",
            f"- Model ladder: **{decision['model_ladder_disposition']}**",
            "",
            f"> {payload['router_limitation']}",
            "",
        ]
    )
    return "\n".join(lines)
