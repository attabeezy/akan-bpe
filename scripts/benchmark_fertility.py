#!/usr/bin/env python3
"""Run one unified SOMAX tokenizer fertility experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from somax.experiment import ExperimentTokenizer, run_fertility_experiment
from somax.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one SOMAX fertility experiment.")
    parser.add_argument("--experiment-id", required=True, help="Stable identifier for this experiment run.")
    parser.add_argument("--control-tokenizer", required=True, help="Baseline tokenizer reference.")
    parser.add_argument("--asr-tokenizer", required=True, help="ASR tokenizer path.")
    parser.add_argument("--tts-tokenizer", required=True, help="TTS tokenizer path.")
    parser.add_argument("--mixed-tokenizer", help="Optional mixed tokenizer path.")
    parser.add_argument("--asr-test-file", required=True, help="ASR JSONL test file.")
    parser.add_argument("--tts-test-file", required=True, help="TTS JSONL test file.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max samples per test set.")
    parser.add_argument("--output", required=True, help="Unified experiment JSON output path.")
    args = parser.parse_args()

    tokenizers = [
        ExperimentTokenizer(name="control", reference=args.control_tokenizer),
        ExperimentTokenizer(name="asr", reference=args.asr_tokenizer),
        ExperimentTokenizer(name="tts", reference=args.tts_tokenizer),
    ]
    if args.mixed_tokenizer:
        tokenizers.append(ExperimentTokenizer(name="mixed", reference=args.mixed_tokenizer))

    payload = run_fertility_experiment(
        experiment_id=args.experiment_id,
        tokenizers=tokenizers,
        asr_test_file=args.asr_test_file,
        tts_test_file=args.tts_test_file,
        max_samples=args.max_samples,
    )

    output_path = Path(args.output)
    write_json(output_path, payload)
    print(f"Experiment written to {output_path}")


if __name__ == "__main__":
    main()
