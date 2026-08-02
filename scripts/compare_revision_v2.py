#!/usr/bin/env python3
"""Generate the v1-versus-v2 scientific checkpoint artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.io import ensure_parent_dir, write_json
from akan_bpe.revision_comparison import (
    build_revision_comparison,
    render_revision_comparison_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/asr_split_revision_v1_vs_v2.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("results/asr_split_revision_v1_vs_v2.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_revision_comparison(
        fertility_v1_path=Path("results/tokenizer_fertility_experiment_001.json"),
        fertility_v2_path=Path("results/tokenizer_fertility_revision_v2.json"),
        heuristic_router_v1_path=Path("results/router_asr_benchmark.json"),
        heuristic_router_v2_path=Path("results/router_asr_revision_v2.json"),
        ml_router_v1_path=Path("results/router_ml_asr_benchmark.json"),
        ml_router_v2_path=Path("results/router_ml_asr_revision_v2.json"),
    )
    write_json(args.output_json, payload)
    ensure_parent_dir(args.output_markdown)
    args.output_markdown.write_text(
        render_revision_comparison_markdown(payload),
        encoding="utf-8",
    )
    decision = payload["decision"]
    print(f"Comparison JSON written to {args.output_json}")
    print(f"Comparison Markdown written to {args.output_markdown}")
    print(f"Material change: {decision['material_change']}")  # type: ignore[index]
    print(f"Model ladder: {decision['model_ladder_disposition']}")  # type: ignore[index]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
