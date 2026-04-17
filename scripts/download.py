#!/usr/bin/env python3
"""Download and normalize Twi datasets for SOMAX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from somax.io import write_json, write_jsonl


def _download_asr_split(split: str, limit: int | None) -> list[dict[str, str]]:
    dataset = load_dataset("google/WaxalNLP", "aka_asr", split=split, streaming=True)
    try:
        dataset = dataset.decode(False).remove_columns(["audio"])
    except Exception:
        pass

    rows: list[dict[str, str]] = []
    for index, item in enumerate(dataset):
        text = str(item.get("transcription") or item.get("text") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "id": str(item.get("id") or f"aka_asr_{split}_{index}"),
                "text": text,
                "source": "aka_asr",
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _detect_pristine_text(item: dict[str, object]) -> str:
    for key in ("twi", "tw", "text", "transcription", "sentence"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    translation = item.get("translation")
    if isinstance(translation, dict):
        for key in ("twi", "tw"):
            value = translation.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _download_pristine_rows(limit: int | None) -> list[dict[str, str]]:
    dataset = load_dataset("ghananlpcommunity/pristine-twi-english", split="train", streaming=True)
    rows: list[dict[str, str]] = []
    for index, item in enumerate(dataset):
        text = _detect_pristine_text(item)
        if not text:
            continue
        rows.append(
            {
                "id": f"pristine_twi_{index}",
                "text": text,
                "source": "pristine_twi",
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _split_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    train_end = int(len(rows) * 0.90)
    validation_end = int(len(rows) * 0.95)
    return {
        "train": rows[:train_end],
        "validation": rows[train_end:validation_end],
        "test": rows[validation_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Twi datasets for SOMAX.")
    parser.add_argument("--output-dir", default="data", help="Directory for normalized JSONL files.")
    parser.add_argument("--asr-limit", type=int, default=None, help="Optional limit per ASR split.")
    parser.add_argument(
        "--tts-limit",
        type=int,
        default=188000,
        help="Optional cap for pristine Twi rows before splitting.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {"language": "twi", "files": {}}

    for split in ("train", "validation", "test"):
        rows = _download_asr_split(split, args.asr_limit)
        path = output_dir / f"aka_asr_{split}.jsonl"
        count = write_jsonl(path, rows)
        manifest["files"][path.name] = {"count": count, "source": "aka_asr"}
        print(f"Wrote {count} rows to {path}")

    pristine_rows = _download_pristine_rows(args.tts_limit)
    for split, rows in _split_rows(pristine_rows).items():
        path = output_dir / f"pristine_twi_{split}.jsonl"
        count = write_jsonl(path, rows)
        manifest["files"][path.name] = {"count": count, "source": "pristine_twi"}
        print(f"Wrote {count} rows to {path}")

    manifest_path = output_dir / "download_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
