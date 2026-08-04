#!/usr/bin/env python3
"""Build the frozen Qwen 0.6B vocabulary-extension tokenizer and metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.io import write_json
from akan_bpe.vocab_extension import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_METADATA_OUTPUT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_TOKENIZER,
    build_extension_from_pretrained,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--candidate-tokenizer", type=Path, default=DEFAULT_SOURCE_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Override the base config hidden size; normally inferred from the model config.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the base tokenizer to already exist in the local Hugging Face cache.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_extension_from_pretrained(
        base_model_id=args.base_model_id,
        candidate_path=args.candidate_tokenizer,
        output_dir=args.output_dir,
        local_files_only=args.local_files_only,
        hidden_size=args.hidden_size,
        untied_embeddings=True,
    )
    write_json(args.metadata_output, payload)
    selection = payload["selection"]
    output = payload["output"]
    print(f"Extension tokenizer written to {output['directory']}")  # type: ignore[index]
    print(
        f"Added {selection['added_token_count']} novel tokens; "  # type: ignore[index]
        f"excluded {selection['exact_collision_count']} exact collisions. "  # type: ignore[index]
    )
    print(f"Metadata written to {args.metadata_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
