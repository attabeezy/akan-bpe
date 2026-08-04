#!/usr/bin/env python3
"""Validate, inspect, or execute one frozen revision-v2 GPU matrix run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.io import write_json
from akan_bpe.model_integration import run_model_integration
from akan_bpe.revision_gpu import (
    RevisionRunSpec,
    load_revision_matrix,
    model_config_for_run,
    result_status,
    validate_run_result,
)

DEFAULT_MATRIX = Path("config/revision_gpu_matrix.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List all expanded run IDs and paths.")
    subparsers.add_parser("status", help="Report completed, invalid, and pending runs.")
    subparsers.add_parser("validate", help="Validate matrix structure and local tokenizer inputs.")

    run_parser = subparsers.add_parser("run", help="Execute exactly one GPU run.")
    selection = run_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id")
    selection.add_argument("--next", action="store_true", help="Run the first missing artifact.")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing result artifact for the selected run.",
    )
    return parser.parse_args()


def _select_run(runs: tuple[RevisionRunSpec, ...], run_id: str | None) -> RevisionRunSpec:
    if run_id is None:
        for run in runs:
            if not run.result_path.exists():
                return run
        raise SystemExit("All matrix result artifacts already exist.")
    for run in runs:
        if run.run_id == run_id:
            return run
    raise SystemExit(f"Unknown matrix run ID: {run_id}")


def main() -> None:
    args = parse_args()
    matrix = load_revision_matrix(args.matrix)

    if args.command == "list":
        for run in matrix.runs:
            print(f"{run.run_id}\t{run.result_path}")
        return

    if args.command == "validate":
        missing_inputs = sorted(
            {str(run.tokenizer_path) for run in matrix.runs if not run.tokenizer_path.exists()}
        )
        if missing_inputs:
            raise SystemExit("Missing tokenizer input(s): " + ", ".join(missing_inputs))
        print(
            f"Matrix {matrix.payload['matrix_id']} is valid: "
            f"{len(matrix.runs)} runs, sha256={matrix.sha256}"
        )
        return

    if args.command == "status":
        status = result_status(matrix)
        invalid: dict[str, str] = {}
        for run in matrix.runs:
            if not run.result_path.exists():
                continue
            payload = json.loads(run.result_path.read_text(encoding="utf-8"))
            try:
                validate_run_result(matrix, run, payload)
            except ValueError as exc:
                invalid[run.run_id] = str(exc)
        print(
            json.dumps(
                {
                    "matrix_id": matrix.payload["matrix_id"],
                    "matrix_sha256": matrix.sha256,
                    "completed": [
                        run_id for run_id in status["completed"] if run_id not in invalid
                    ],
                    "invalid": invalid,
                    "pending": status["pending"],
                },
                indent=2,
            )
        )
        return

    run = _select_run(matrix.runs, args.run_id)
    config = model_config_for_run(matrix, run)
    if run.result_path.exists() and not args.force:
        payload = json.loads(run.result_path.read_text(encoding="utf-8"))
        validate_run_result(matrix, run, payload)
        print(f"Already complete and valid: {run.run_id}")
        return
    if args.dry_run:
        print(json.dumps({"run": run.run_id, "config": config.__dict__}, default=str, indent=2))
        return

    payload = run_model_integration(config)
    payload["revision_matrix"] = {
        "matrix_id": matrix.payload["matrix_id"],
        "sha256": matrix.sha256,
        "source": str(matrix.path).replace("\\", "/"),
        "goals": list(run.goals),
    }
    validate_run_result(matrix, run, payload)
    write_json(run.result_path, payload)
    print(f"Completed and validated: {run.run_id}")
    print(f"Result: {run.result_path}")
    print(f"Checkpoint: {run.output_dir}")


if __name__ == "__main__":
    main()
