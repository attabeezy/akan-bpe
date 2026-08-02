from __future__ import annotations

import json
from pathlib import Path

import pytest

from akan_bpe.revision_data import build_filtered_jsonl_revision, normalized_text_sha256
from akan_bpe.revision_manifest import sha256_file


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_build_filtered_revision_removes_only_match_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "historical.jsonl"
    output = tmp_path / "revision.jsonl"
    correction = tmp_path / "correction.json"
    rows = [
        {"id": "one", "text": "Keep me", "source": "fixture"},
        {"id": "two", "text": "Leaked text", "source": "fixture"},
        {"id": "three", "text": "Keep me too", "source": "fixture"},
    ]
    _write_rows(source, rows)
    original_bytes = source.read_bytes()

    payload = build_filtered_jsonl_revision(
        source_path=source,
        output_path=output,
        correction_path=correction,
        remove_normalized_text_sha256=normalized_text_sha256(" leaked   TEXT "),
        reason="Fixture correction.",
        expected_source_sha256=sha256_file(source),
    )

    assert source.read_bytes() == original_bytes
    assert [json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()] == [
        "one",
        "three",
    ]
    assert payload["source"]["rows"] == 3  # type: ignore[index]
    assert payload["output"]["rows"] == 2  # type: ignore[index]
    assert payload["removed_rows"][0]["id"] == "two"  # type: ignore[index]
    assert json.loads(correction.read_text(encoding="utf-8")) == payload


def test_build_filtered_revision_rejects_source_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "historical.jsonl"
    _write_rows(source, [{"id": "one", "text": "Leaked", "source": "fixture"}])

    with pytest.raises(ValueError, match="must not overwrite"):
        build_filtered_jsonl_revision(
            source_path=source,
            output_path=source,
            correction_path=tmp_path / "correction.json",
            remove_normalized_text_sha256=normalized_text_sha256("Leaked"),
            reason="Fixture correction.",
        )


def test_build_filtered_revision_requires_exactly_one_match(tmp_path: Path) -> None:
    source = tmp_path / "historical.jsonl"
    _write_rows(
        source,
        [
            {"id": "one", "text": "Duplicate", "source": "fixture"},
            {"id": "two", "text": " duplicate ", "source": "fixture"},
        ],
    )

    with pytest.raises(ValueError, match="exactly one row"):
        build_filtered_jsonl_revision(
            source_path=source,
            output_path=tmp_path / "revision.jsonl",
            correction_path=tmp_path / "correction.json",
            remove_normalized_text_sha256=normalized_text_sha256("Duplicate"),
            reason="Fixture correction.",
        )


def test_build_filtered_revision_requires_frozen_source_hash(tmp_path: Path) -> None:
    source = tmp_path / "historical.jsonl"
    _write_rows(source, [{"id": "one", "text": "Leaked", "source": "fixture"}])

    with pytest.raises(ValueError, match="does not match"):
        build_filtered_jsonl_revision(
            source_path=source,
            output_path=tmp_path / "revision.jsonl",
            correction_path=tmp_path / "correction.json",
            remove_normalized_text_sha256=normalized_text_sha256("Leaked"),
            reason="Fixture correction.",
            expected_source_sha256="0" * 64,
        )
