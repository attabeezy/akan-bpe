#!/usr/bin/env python3
"""Validate and aggregate the complete revision-v2 GPU run matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.io import write_json
from akan_bpe.revision_gpu import aggregate_matrix_results, load_revision_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("config/revision_gpu_matrix.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_revision_matrix(args.matrix)
    output = args.output or Path(matrix.payload["paths"]["aggregate_result"])
    payload = aggregate_matrix_results(matrix)
    write_json(output, payload)
    print(f"Validated {payload['run_count']} runs and wrote {output}")


if __name__ == "__main__":
    main()
