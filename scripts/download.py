#!/usr/bin/env python3
"""Download and normalize Akan datasets for Akan-BPE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.io import write_json, write_jsonl


def _download_asr_split(split: str, limit: int | None) -> list[dict[str, str]]:
    dataset = load_dataset("google/WaxalNLP", "aka_asr", split=split, streaming=True)
    # Select only the text columns up-front so the audio bytes are never fetched.
    try:
        dataset = dataset.select_columns(["transcription", "id"])
    except Exception:
        try:
            dataset = dataset.remove_columns(["audio"])
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
    train_end = int(len(rows) * 0.80)
    validation_end = int(len(rows) * 0.90)
    return {
        "train": rows[:train_end],
        "validation": rows[train_end:validation_end],
        "test": rows[validation_end:],
    }


def _assert_healthy_split(source: str, splits: dict[str, list[dict[str, str]]]) -> None:
    """Fail loudly if a split looks truncated (e.g. a 1-row test set).

    A streaming download that drops the connection mid-fetch can leave a tiny,
    statistically useless validation/test split while train still looks large.
    For any reasonably sized corpus the 10% holdouts must have a sane floor; a
    1-sample test set is the stale-download artifact we want to catch here.
    """
    total = sum(len(rows) for rows in splits.values())
    if total < 100:
        return
    floor = 10
    for name in ("validation", "test"):
        if len(splits[name]) < floor:
            raise ValueError(
                f"{source}: {name} split has only {len(splits[name])} rows out of {total} "
                f"total (expected >= {floor}). The stream was likely truncated; re-run the "
                "download with a stable connection."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Akan datasets for Akan-BPE.")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for normalized JSONL files.",
    )
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

    manifest: dict[str, object] = {"language": "akan", "files": {}}

    asr_rows = _download_asr_split("train", args.asr_limit)
    asr_splits = _split_rows(asr_rows)
    _assert_healthy_split("aka_asr", asr_splits)
    for split, rows in asr_splits.items():
        path = output_dir / f"aka_asr_{split}.jsonl"
        count = write_jsonl(path, rows)
        manifest["files"][path.name] = {"count": count, "source": "aka_asr"}
        print(f"Wrote {count} rows to {path}")

    pristine_rows = _download_pristine_rows(args.tts_limit)
    pristine_splits = _split_rows(pristine_rows)
    _assert_healthy_split("pristine_twi", pristine_splits)
    for split, rows in pristine_splits.items():
        path = output_dir / f"pristine_twi_{split}.jsonl"
        count = write_jsonl(path, rows)
        manifest["files"][path.name] = {"count": count, "source": "pristine_twi"}
        print(f"Wrote {count} rows to {path}")

    manifest_path = output_dir / "download_manifest.json"
    write_json(manifest_path, manifest)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
