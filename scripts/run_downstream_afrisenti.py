#!/usr/bin/env python3
"""Validate, execute, resume, and aggregate the frozen AfriSenti matrix."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.downstream import (
    DownstreamManifest,
    DownstreamRunSpec,
    aggregate_results,
    audit_adaptation_overlap,
    audit_dataset_splits,
    evaluate_model,
    evaluate_predictions,
    fetch_dataset,
    load_downstream_manifest,
    render_markdown_table,
    resolve_demonstrations,
    validate_demonstration_selection,
    validate_result,
)
from akan_bpe.io import ensure_parent_dir, write_json
from akan_bpe.model_integration import (
    load_saved_qwen_artifacts,
    run_model_integration,
)
from akan_bpe.revision_gpu import (
    load_revision_matrix,
    model_config_for_run,
    validate_run_result,
)

DEFAULT_MANIFEST = Path("config/downstream_afrisenti.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate local contracts without network/GPU.")
    subparsers.add_parser(
        "fetch-data", help="Download and validate pinned AfriSenti parquet files."
    )
    subparsers.add_parser("list", help="List all expanded result IDs.")
    subparsers.add_parser("status", help="Validate present results and report pending runs.")
    run_parser = subparsers.add_parser("run", help="Recreate/evaluate exactly one run.")
    selection = run_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--next", action="store_true")
    selection.add_argument("--run-id")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("aggregate", help="Validate all results and generate aggregate/table.")
    return parser.parse_args()


def validate_contract(manifest: DownstreamManifest) -> None:
    matrix_path = Path(manifest.payload["source_matrix"]["path"])
    matrix = load_revision_matrix(matrix_path)
    source = manifest.payload["source_matrix"]
    if matrix.sha256 != source["sha256"] or matrix.payload["matrix_id"] != source["matrix_id"]:
        raise ValueError("Frozen source matrix identity does not match downstream manifest.")
    selected = {run.source_run_id for run in manifest.runs if run.source_run_id}
    matrix_ids = {run.run_id for run in matrix.runs}
    if not selected <= matrix_ids:
        raise ValueError("Downstream manifest references unknown revision-v2 runs.")
    for run in matrix.runs:
        if run.run_id not in selected:
            continue
        if not run.result_path.exists():
            raise ValueError(f"Missing canonical source result {run.result_path}.")
        payload = json.loads(run.result_path.read_text(encoding="utf-8"))
        validate_run_result(matrix, run, payload)
        if not run.tokenizer_path.exists():
            raise ValueError(f"Missing tokenizer for downstream arm: {run.tokenizer_path}")


def select_run(manifest: DownstreamManifest, run_id: str | None) -> DownstreamRunSpec:
    if run_id:
        for run in manifest.runs:
            if run.run_id == run_id:
                return run
        raise SystemExit(f"Unknown downstream run ID: {run_id}")
    for run in manifest.runs:
        if not run.result_path.exists():
            return run
    raise SystemExit("All downstream runs are complete.")


def load_base_model(model_id: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization, dtype="auto"
    )
    model.eval()
    return model, tokenizer


def recreate_adapted_model(
    manifest: DownstreamManifest, run: DownstreamRunSpec
) -> tuple[Any, Any, dict[str, Any], Path]:
    matrix = load_revision_matrix(Path(manifest.payload["source_matrix"]["path"]))
    source_run = next(item for item in matrix.runs if item.run_id == run.source_run_id)
    canonical = json.loads(source_run.result_path.read_text(encoding="utf-8"))
    validate_run_result(matrix, source_run, canonical)
    temporary_dir = Path(manifest.payload["paths"]["temporary_models"]) / run.run_id
    if temporary_dir.exists():
        raise RuntimeError(
            f"Preserved temporary checkpoint from an earlier failed run: {temporary_dir}. "
            "Inspect it, then remove that exact directory before retrying."
        )
    config = replace(
        model_config_for_run(matrix, source_run),
        output_dir=temporary_dir.as_posix(),
        results_output="",
        compute_base_bpb=False,
        generation_samples=1,
        generation_eval_samples=0,
    )
    recreation = run_model_integration(config)
    recreation_eval = recreation["eval"]
    if not isinstance(recreation_eval, dict):
        raise ValueError("Checkpoint recreation returned invalid evaluation metadata.")
    model, tokenizer = load_saved_qwen_artifacts(config, run.model_id)
    metadata = {
        "source_matrix_sha256": matrix.sha256,
        "source_result_path": source_run.result_path.as_posix(),
        "source_result_bpb": canonical["eval"]["bpb"]["experiment"]["bits_per_byte"],
        "recreated_bpb": recreation_eval["bpb"]["experiment"]["bits_per_byte"],
        "training": recreation["training"],
        "reload_verification": recreation["reload_verification"],
    }
    return model, tokenizer, metadata, temporary_dir


def execute_run(manifest: DownstreamManifest, run: DownstreamRunSpec) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Downstream model execution requires a CUDA GPU.")
    splits = fetch_dataset(manifest)
    audit = audit_dataset_splits(splits, manifest)
    validate_demonstration_selection(splits, manifest)
    adaptation_overlap = audit_adaptation_overlap(
        splits["test"],
        [Path(path) for path in manifest.payload["dataset"]["adaptation_overlap_files"]],
    )
    for required in ("data/pristine_twi_train.jsonl", "data/pristine_twi_test.jsonl"):
        if adaptation_overlap.get(required) != 0:
            raise ValueError(f"Expected zero AfriSenti overlap with required corpus {required}.")
    demonstrations = resolve_demonstrations(splits["train"], manifest)
    temporary_dir: Path | None = None
    recreation: dict[str, Any] | None = None
    if run.strategy == "original":
        model, tokenizer = load_base_model(run.model_id)
    else:
        model, tokenizer, recreation, temporary_dir = recreate_adapted_model(manifest, run)
    try:
        predictions = evaluate_model(model, tokenizer, splits["test"], demonstrations, manifest)
        metrics = evaluate_predictions(predictions, manifest, audit["clean_test_indices"])
        payload = {
            "schema_version": 1,
            "run_id": run.run_id,
            "model_id": run.model_id,
            "strategy": run.strategy,
            "initialization": run.initialization,
            "seed": run.seed,
            "source_run_id": run.source_run_id,
            "manifest_sha256": manifest.sha256,
            "dataset_revision": manifest.payload["dataset"]["revision"],
            "dataset_audit": audit,
            "adaptation_corpus_overlap": adaptation_overlap,
            "prompt": {
                "demonstrations": manifest.payload["prompt"]["demonstrations"],
                "cyclic_orders": manifest.payload["prompt"]["cyclic_orders"],
                "label_codes": manifest.payload["prompt"]["label_codes"],
            },
            "device": {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "torch": torch.__version__,
                "python": platform.python_version(),
            },
            "checkpoint_recreation": recreation,
            "predictions": predictions,
            "metrics": metrics,
        }
        validate_result(manifest, run, payload)
        write_json(run.result_path, payload)
        validate_result(manifest, run, json.loads(run.result_path.read_text(encoding="utf-8")))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if temporary_dir is not None:
        shutil.rmtree(temporary_dir)
        print(f"Removed validated temporary checkpoint: {temporary_dir}")


def main() -> None:
    args = parse_args()
    manifest = load_downstream_manifest(args.manifest)
    validate_contract(manifest)
    if args.command == "validate":
        print(f"Valid downstream manifest: {len(manifest.runs)} runs, sha256={manifest.sha256}")
        return
    if args.command == "fetch-data":
        splits = fetch_dataset(manifest)
        audit = audit_dataset_splits(splits, manifest)
        validate_demonstration_selection(splits, manifest)
        adaptation_overlap = audit_adaptation_overlap(
            splits["test"],
            [Path(path) for path in manifest.payload["dataset"]["adaptation_overlap_files"]],
        )
        resolve_demonstrations(splits["train"], manifest)
        print(
            json.dumps(
                {
                    "counts": audit["counts"],
                    "overlaps": audit["overlaps"],
                    "overlap_label_conflicts": audit["overlap_label_conflicts"],
                    "within_split_label_conflicts": audit["within_split_label_conflicts"],
                    "clean_test_rows": len(audit["clean_test_indices"]),
                    "adaptation_corpus_overlap": adaptation_overlap,
                },
                indent=2,
            )
        )
        return
    if args.command == "list":
        for run in manifest.runs:
            print(f"{run.run_id}\t{run.result_path}")
        return
    if args.command == "status":
        complete, pending, invalid = [], [], {}
        for run in manifest.runs:
            if not run.result_path.exists():
                pending.append(run.run_id)
                continue
            try:
                validate_result(
                    manifest, run, json.loads(run.result_path.read_text(encoding="utf-8"))
                )
                complete.append(run.run_id)
            except ValueError as exc:
                invalid[run.run_id] = str(exc)
        print(json.dumps({"complete": complete, "pending": pending, "invalid": invalid}, indent=2))
        return
    if args.command == "aggregate":
        aggregate = aggregate_results(manifest)
        output = Path(manifest.payload["paths"]["aggregate_result"])
        table = Path(manifest.payload["paths"]["generated_table"])
        write_json(output, aggregate)
        ensure_parent_dir(table)
        table.write_text(render_markdown_table(aggregate), encoding="utf-8")
        print(f"Wrote {output} and {table}")
        return
    run = select_run(manifest, args.run_id)
    if run.result_path.exists() and not args.force:
        validate_result(manifest, run, json.loads(run.result_path.read_text(encoding="utf-8")))
        print(f"Already complete and valid: {run.run_id}")
        return
    if args.dry_run:
        print(json.dumps(run.__dict__, default=str, indent=2))
        return
    execute_run(manifest, run)
    print(f"Completed and validated: {run.run_id}")


if __name__ == "__main__":
    main()
