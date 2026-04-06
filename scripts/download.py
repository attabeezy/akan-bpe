#!/usr/bin/env python3
"""Download WAXAL dataset for African language tokenization research.

Target languages: Akan (proof-of-concept)
Dataset: google/WaxalNLP from HuggingFace
Dataset splits: ASR (spontaneous), TTS (formal)

Usage:
    python scripts/00_download_data.py --lang akan --output data/

Language configs:
    - akan: aka_asr + twi_tts
    - yoruba: yor_tts (TTS only, ASR not available)
    - swahili: swa_tts (TTS only)
"""

import argparse
import json
from pathlib import Path

DATASET_CONFIGS = {
    "akan": {"asr": "aka_asr", "tts": "twi_tts"},
    "yoruba": {"asr": None, "tts": "yor_tts"},
    "swahili": {"asr": None, "tts": "swa_tts"},
}


def download_split(config_name: str, split: str, output_dir: Path) -> dict:
    """Download a specific dataset split.

    Args:
        config_name: HuggingFace config name (e.g., 'aka_asr').
        split: Split name ('train', 'validation', 'test').
        output_dir: Directory to save data.

    Returns:
        Metadata about downloaded split.
    """
    from datasets import load_dataset

    print(f"Downloading {config_name}/{split}...")
    dataset = load_dataset("google/WaxalNLP", config_name, split=split, trust_remote_code=True)

    output_file = output_dir / f"{config_name}_{split}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            item_dict = {
                "id": item["id"],
                "speaker_id": item.get("speaker_id", ""),
                "transcription": item.get("transcription", item.get("text", "")),
                "language": item.get("language", item.get("locale", "")),
                "gender": item.get("gender", ""),
            }
            f.write(json.dumps(item_dict, ensure_ascii=False) + "\n")

    return {"file": str(output_file), "count": len(dataset)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WAXAL dataset")
    parser.add_argument("--lang", type=str, default="akan", choices=["akan", "yoruba", "swahili"])
    parser.add_argument("--output", type=str, default="data/")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()

    output_dir = Path(args.output) / args.lang
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DATASET_CONFIGS[args.lang]
    metadata = {"language": args.lang, "asr": {}, "tts": {}}

    print(f"Downloading WAXAL-{args.lang} dataset...")
    print(f"Target directory: {output_dir}")

    for split_type in ["asr", "tts"]:
        config_name = config[split_type]
        if config_name is None:
            print(f"  No {split_type.upper()} config for {args.lang}")
            continue

        print(f"\n{split_type.upper()} config: {config_name}")
        for split in args.splits:
            try:
                result = download_split(config_name, split, output_dir)
                metadata[split_type][split] = result
                print(f"  {split}: {result['count']} samples -> {result['file']}")
            except Exception as e:
                print(f"  {split}: Error - {e}")

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {metadata_file}")
    print("Download complete!")


if __name__ == "__main__":
    main()
