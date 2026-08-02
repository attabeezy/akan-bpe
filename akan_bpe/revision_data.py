"""Build and verify corrected revision datasets without mutating historical inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from akan_bpe.io import ensure_parent_dir, write_json
from akan_bpe.revision_manifest import normalize_text, sha256_file


def normalized_text_sha256(text: str) -> str:
    """Hash text after applying the revision leakage normalization."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def build_filtered_jsonl_revision(
    *,
    source_path: Path,
    output_path: Path,
    correction_path: Path,
    remove_normalized_text_sha256: str,
    reason: str,
    expected_source_sha256: str | None = None,
) -> dict[str, object]:
    """Copy a JSONL dataset while removing exactly one normalized-text match."""
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Revision output must not overwrite the historical source file.")
    source_sha256 = sha256_file(source_path)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ValueError(
            "Historical source hash does not match the expected frozen artifact: "
            f"{source_sha256} != {expected_source_sha256}"
        )
    if not reason.strip():
        raise ValueError("A non-empty correction reason is required.")

    source_lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept_lines: list[str] = []
    removed: list[dict[str, object]] = []
    source_rows = 0
    for line_number, line in enumerate(source_lines, start=1):
        if not line.strip():
            kept_lines.append(line)
            continue
        source_rows += 1
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Source row {line_number} must be a JSON object.")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Source row {line_number} has no non-empty text field.")
        digest = normalized_text_sha256(text)
        if digest == remove_normalized_text_sha256:
            removed.append(
                {
                    "source_row_number": line_number,
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "normalized_text_sha256": digest,
                    "text": text,
                }
            )
        else:
            kept_lines.append(line)

    if len(removed) != 1:
        raise ValueError(
            "Revision correction must remove exactly one row; "
            f"found {len(removed)} matches for {remove_normalized_text_sha256}."
        )

    ensure_parent_dir(output_path)
    output_path.write_text("".join(kept_lines), encoding="utf-8", newline="")
    output_rows = source_rows - 1
    payload: dict[str, object] = {
        "schema_version": 1,
        "operation": "remove_normalized_text_match",
        "reason": reason,
        "source": {
            "path": source_path.as_posix(),
            "rows": source_rows,
            "bytes": source_path.stat().st_size,
            "sha256": source_sha256,
        },
        "output": {
            "path": output_path.as_posix(),
            "rows": output_rows,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "removed_rows": removed,
    }
    write_json(correction_path, payload)
    return payload
