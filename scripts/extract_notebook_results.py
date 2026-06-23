from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_NOTEBOOKS = (
    Path("notebooks/run-full-light.ipynb"),
    Path("notebooks/run-full-heavy.ipynb"),
)
DEFAULT_OUTPUT = Path("results/notebook-ladder-results.json")
ARM_BEGIN = "BEGIN_FULL_JSON"
ARM_END = "END_FULL_JSON"


class NotebookResultError(RuntimeError):
    """Raised when an executed notebook does not contain a usable result block."""


def _output_text(cell: dict[str, Any]) -> str:
    parts: list[str] = []
    for output in cell.get("outputs", []):
        text = output.get("text", "")
        if isinstance(text, list):
            parts.append("".join(text))
        elif text:
            parts.append(str(text))
    return "".join(parts)


def _split_from_notebook_path(path: Path) -> str:
    stem = path.stem
    prefix = "run-full-"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _model_slug_from_experiment_id(experiment_id: str) -> str:
    prefix = "run-"
    suffix = "-mixed"
    mean_suffix = "-meansub"
    slug = experiment_id
    if slug.startswith(prefix):
        slug = slug[len(prefix) :]
    if slug.endswith(mean_suffix):
        slug = slug[: -len(mean_suffix)]
    if slug.endswith(suffix):
        slug = slug[: -len(suffix)]
    return slug


def _arm_from_payload(payload: dict[str, Any]) -> str:
    mode = payload.get("embedding_init_mode")
    if mode in {"random", "mean_subword"}:
        return str(mode)

    experiment_id = str(payload.get("experiment_id", ""))
    if experiment_id.endswith("-meansub"):
        return "mean_subword"
    return "random"


def _summary_from_arm(payload: dict[str, Any]) -> dict[str, Any]:
    counts = payload.get("token_count_comparison") or {}
    base_counts = counts.get("base_model_tokenizer") or {}
    akan_counts = counts.get("experiment_tokenizer") or {}
    eval_payload = payload.get("eval") or {}
    bpb = eval_payload.get("bpb") or {}
    experiment_bpb = bpb.get("experiment") or {}
    base_bpb = bpb.get("base") or {}
    generation_quality = eval_payload.get("generation_quality") or {}

    return {
        "eval_loss": eval_payload.get("eval_loss"),
        "perplexity": eval_payload.get("perplexity"),
        "experiment_bpb": experiment_bpb.get("bits_per_byte"),
        "base_bpb": base_bpb.get("bits_per_byte"),
        "bpb_improvement": bpb.get("improvement"),
        "base_fertility": base_counts.get("fertility"),
        "akan_fertility": akan_counts.get("fertility"),
        "token_reduction_ratio": counts.get("token_reduction_ratio"),
        "generation_quality": {
            "chrf": generation_quality.get("chrf"),
            "chrfpp": generation_quality.get("chrfpp"),
            "num_examples": generation_quality.get("num_examples"),
            "prompt_words": generation_quality.get("prompt_words"),
            "reference_words": generation_quality.get("reference_words"),
            "max_new_tokens": generation_quality.get("max_new_tokens"),
        },
    }


def _extract_json_between(text: str, json_start: int) -> tuple[dict[str, Any], int]:
    decoder = json.JSONDecoder()
    try:
        payload, offset = decoder.raw_decode(text[json_start:])
    except json.JSONDecodeError as exc:
        raise NotebookResultError(f"Malformed notebook result JSON: {exc}") from exc
    return payload, json_start + offset


def _extract_arm_payloads(text: str, path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    start = 0
    while True:
        begin = text.find(ARM_BEGIN, start)
        if begin < 0:
            break

        end = text.find(ARM_END, begin)
        if end < 0:
            raise NotebookResultError(f"No {ARM_END} marker found after arm block in {path}")

        json_start = text.find("{", begin, end)
        if json_start < 0:
            raise NotebookResultError(f"No JSON object found in arm result block: {path}")

        try:
            payload = json.loads(text[json_start:end].strip())
        except json.JSONDecodeError:
            payload, _ = _extract_json_between(text, json_start)
        payloads.append(payload)
        start = end + len(ARM_END)

    return payloads


def _build_payload_from_arm_blocks(path: Path, text: str) -> dict[str, Any]:
    arm_payloads = _extract_arm_payloads(text, path)
    if not arm_payloads:
        raise NotebookResultError(
            f"No BEGIN_NOTEBOOK_FULL_JSON or {ARM_BEGIN} blocks found in {path}"
        )

    split = _split_from_notebook_path(path)
    model_order: list[str] = []
    summary: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, Any]] = {}

    for arm_payload in arm_payloads:
        experiment_id = arm_payload.get("experiment_id")
        if not experiment_id:
            raise NotebookResultError(f"Arm result block in {path} is missing experiment_id")

        model_slug = _model_slug_from_experiment_id(str(experiment_id))
        arm = _arm_from_payload(arm_payload)

        if model_slug not in summary:
            model_order.append(model_slug)
            summary[model_slug] = {}
            runs[model_slug] = {}
        if arm in runs[model_slug]:
            raise NotebookResultError(f"Duplicate arm found in {path}: {model_slug}/{arm}")

        summary[model_slug][arm] = _summary_from_arm(arm_payload)
        runs[model_slug][arm] = arm_payload

    return {
        "split": split,
        "model_slugs": model_order,
        "summary": summary,
        "runs": runs,
    }


def extract_notebook_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise NotebookResultError(f"Missing notebook: {path}")

    notebook = json.loads(path.read_text(encoding="utf-8"))
    text = "".join(_output_text(cell) for cell in notebook.get("cells", []))
    begin = text.find("BEGIN_NOTEBOOK_FULL_JSON")
    if begin < 0:
        return _build_payload_from_arm_blocks(path, text)

    end = text.find("END_NOTEBOOK_FULL_JSON", begin)
    if end < 0:
        raise NotebookResultError(f"No END_NOTEBOOK_FULL_JSON marker found in {path}")

    json_start = text.find("{", begin, end)
    if json_start < 0:
        raise NotebookResultError(f"No JSON object found in notebook result block: {path}")

    try:
        payload = json.loads(text[json_start:end].strip())
    except json.JSONDecodeError as exc:
        raise NotebookResultError(f"Malformed notebook result JSON in {path}: {exc}") from exc

    required = {"split", "model_slugs", "summary", "runs"}
    missing = sorted(required - payload.keys())
    if missing:
        raise NotebookResultError(f"Notebook result block in {path} is missing keys: {missing}")

    return payload


def _round_or_none(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def flatten_summary(splits: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for split_name, payload in splits.items():
        for model_slug in payload["model_slugs"]:
            arms = payload["summary"].get(model_slug)
            if arms is None:
                raise NotebookResultError(
                    f"Split {split_name!r} lists model {model_slug!r}, but summary is missing it"
                )
            for arm, metrics in arms.items():
                key = (model_slug, arm)
                if key in seen:
                    raise NotebookResultError(f"Duplicate model/arm found: {model_slug}/{arm}")
                seen.add(key)

                generation_quality = metrics.get("generation_quality") or {}
                run = payload["runs"].get(model_slug, {}).get(arm, {})
                rows.append(
                    {
                        "split": split_name,
                        "model": model_slug,
                        "arm": arm,
                        "experiment_id": run.get("experiment_id"),
                        "model_id": run.get("model_id"),
                        "embedding_init_mode": run.get("embedding_init_mode", arm),
                        "eval_loss": _round_or_none(metrics.get("eval_loss")),
                        "perplexity": _round_or_none(metrics.get("perplexity")),
                        "base_fertility": _round_or_none(metrics.get("base_fertility")),
                        "akan_fertility": _round_or_none(metrics.get("akan_fertility")),
                        "token_reduction_ratio": _round_or_none(
                            metrics.get("token_reduction_ratio")
                        ),
                        "base_bpb": _round_or_none(metrics.get("base_bpb")),
                        "experiment_bpb": _round_or_none(metrics.get("experiment_bpb")),
                        "bpb_improvement": _round_or_none(metrics.get("bpb_improvement")),
                        "generation_quality": {
                            "chrf": _round_or_none(generation_quality.get("chrf")),
                            "chrfpp": _round_or_none(generation_quality.get("chrfpp")),
                            "num_examples": generation_quality.get("num_examples"),
                            "prompt_words": generation_quality.get("prompt_words"),
                            "reference_words": generation_quality.get("reference_words"),
                            "max_new_tokens": generation_quality.get("max_new_tokens"),
                        },
                    }
                )

    return rows


def build_interpretation(summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary:
        by_model.setdefault(row["model"], {})[row["arm"]] = row

    chrf_wins: list[str] = []
    chrfpp_wins: list[str] = []
    bpb_wins: list[str] = []
    base_bpb_wins: list[str] = []

    for model, arms in by_model.items():
        random = arms.get("random")
        mean = arms.get("mean_subword")
        if not random or not mean:
            continue

        if mean["generation_quality"]["chrf"] > random["generation_quality"]["chrf"]:
            chrf_wins.append(model)
        if mean["generation_quality"]["chrfpp"] > random["generation_quality"]["chrfpp"]:
            chrfpp_wins.append(model)
        if mean["experiment_bpb"] < random["experiment_bpb"]:
            bpb_wins.append(model)
        if mean["experiment_bpb"] < mean["base_bpb"]:
            base_bpb_wins.append(model)

    total_models = len(by_model)
    return {
        "model_count": total_models,
        "arm_count": len(summary),
        "mean_subword_chrf_wins": f"{len(chrf_wins)}/{total_models}",
        "mean_subword_chrf_win_models": chrf_wins,
        "mean_subword_chrfpp_wins": f"{len(chrfpp_wins)}/{total_models}",
        "mean_subword_chrfpp_win_models": chrfpp_wins,
        "mean_subword_bpb_wins": f"{len(bpb_wins)}/{total_models}",
        "mean_subword_bpb_win_models": bpb_wins,
        "mean_subword_beats_base_bpb": f"{len(base_bpb_wins)}/{total_models}",
        "mean_subword_beats_base_bpb_models": base_bpb_wins,
        "generation_quality_protocol": {
            "num_examples": 512,
            "prompt_words": 48,
            "reference_words": 64,
            "max_new_tokens": 64,
            "metric": "chrF / chrF++ on held-out Twi continuations",
        },
    }


def build_notebook_results(notebooks: list[Path]) -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    seen_models: set[str] = set()

    for path in notebooks:
        payload = extract_notebook_payload(path)
        split_name = payload["split"]
        if split_name in splits:
            raise NotebookResultError(f"Duplicate split found: {split_name}")
        duplicate_models = sorted(seen_models.intersection(payload["model_slugs"]))
        if duplicate_models:
            raise NotebookResultError(f"Duplicate model slug(s) found: {duplicate_models}")
        seen_models.update(payload["model_slugs"])
        splits[split_name] = payload

    summary = flatten_summary(splits)
    return {
        "version": 1,
        "source_of_truth": [str(path) for path in notebooks],
        "description": (
            "Derived from executed notebook output blocks. The notebooks are the authoritative "
            "source for these ladder and generation-quality results."
        ),
        "splits": splits,
        "runs": {
            model: arms
            for payload in splits.values()
            for model, arms in payload["runs"].items()
        },
        "summary": summary,
        "interpretation": build_interpretation(summary),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract consolidated ladder results from executed notebook JSON blocks."
    )
    parser.add_argument(
        "--notebooks",
        nargs="+",
        type=Path,
        default=list(DEFAULT_NOTEBOOKS),
        help="Executed notebook paths to parse.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the consolidated JSON artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_notebook_results(args.notebooks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
