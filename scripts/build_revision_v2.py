#!/usr/bin/env python3
"""Build the leak-free ASR revision-v2 test split without changing historical data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.revision_data import build_filtered_jsonl_revision

HISTORICAL_ASR_TEST_SHA256 = "4a682f1f8b8ec9c65e8d5f5c7d99aa592c033959401a387516dc3400acdc488b"
LEAKED_NORMALIZED_TEXT_SHA256 = (
    "8eaa78663e83daf1e9eaf897af2468f67580636041af808160d4927a1fc7e2e6"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/aka_asr_test.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/revision_v2/aka_asr_test.jsonl"),
    )
    parser.add_argument(
        "--correction-output",
        type=Path,
        default=Path("data/revision_v2/asr_test_correction.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_filtered_jsonl_revision(
        source_path=args.source,
        output_path=args.output,
        correction_path=args.correction_output,
        remove_normalized_text_sha256=LEAKED_NORMALIZED_TEXT_SHA256,
        reason=(
            "Remove the sole normalized-text overlap with the frozen ASR training split; "
            "retain the 1,011-row source as historical v1 evidence."
        ),
        expected_source_sha256=HISTORICAL_ASR_TEST_SHA256,
    )
    print(
        f"Wrote {payload['output']['rows']} rows to {payload['output']['path']} "  # type: ignore[index]
        f"from {payload['source']['rows']} historical rows."  # type: ignore[index]
    )
    print(f"Correction record written to {args.correction_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
