#!/usr/bin/env python3
"""Generate the frozen revision-v2 router audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.router_audit import build_router_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/router_audit_revision_v2.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_router_audit(
        classifier_path=Path("models/router_classifier.pkl"),
        asr_train_path=Path("data/aka_asr_train.jsonl"),
        tts_train_path=Path("data/pristine_twi_train.jsonl"),
        asr_test_path=Path("data/revision_v2/aka_asr_test.jsonl"),
        tts_test_path=Path("data/pristine_twi_test.jsonl"),
        heuristic_asr_result_path=Path("results/router_asr_revision_v2.json"),
        heuristic_tts_result_path=Path("results/router_tts_benchmark.json"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Router audit written to {args.output}")


if __name__ == "__main__":
    main()
