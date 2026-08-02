#!/usr/bin/env python3
"""Validate the frozen Akan-BPE revision experiment manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.revision_manifest import validate_revision_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("config/revision_manifest.yaml"),
        help="Repository-relative or absolute manifest path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_revision_manifest(args.manifest, repo_root=ROOT)
    if errors:
        print(f"Revision manifest validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Revision manifest is valid: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
