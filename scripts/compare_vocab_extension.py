#!/usr/bin/env python3
"""Evaluate original, extension, and replacement tokenizers on frozen revision-v2 data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.io import write_json
from akan_bpe.vocab_extension import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_METADATA_OUTPUT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_TOKENIZER,
    build_intrinsic_extension_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--extension-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replacement-tokenizer", type=Path, default=DEFAULT_SOURCE_TOKENIZER)
    parser.add_argument("--extension-metadata", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument(
        "--asr-test",
        type=Path,
        default=Path("data/revision_v2/aka_asr_test.jsonl"),
    )
    parser.add_argument(
        "--formal-test",
        type=Path,
        default=Path("data/pristine_twi_test.jsonl"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/vocab_extension_intrinsic_revision_v2.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_intrinsic_extension_comparison(
        base_model_id=args.base_model_id,
        extension_dir=args.extension_dir,
        replacement_path=args.replacement_tokenizer,
        extension_metadata_path=args.extension_metadata,
        asr_test_path=args.asr_test,
        formal_test_path=args.formal_test,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        local_files_only=True,
    )
    write_json(args.output, payload)
    print(f"Intrinsic extension comparison written to {args.output}")
    comparisons = cast(dict[str, dict[str, float]], payload["comparisons"])
    for regime, result in comparisons.items():
        print(
            f"{regime}: original={result['original_fertility']:.6f}, "
            f"extension={result['extension_fertility']:.6f}, "
            f"replacement={result['replacement_fertility']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
