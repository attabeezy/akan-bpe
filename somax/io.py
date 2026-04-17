"""Low-level file helpers for SOMAX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    """Write a JSON payload with stable formatting."""
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    """Write JSONL rows and return the number written."""
    ensure_parent_dir(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
