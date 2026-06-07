from __future__ import annotations

import pytest

from scripts.download import _assert_healthy_split, _split_rows


def _rows(n: int) -> list[dict[str, str]]:
    return [{"id": str(i), "text": f"text{i}", "source": "aka_asr"} for i in range(n)]


def test_split_rows_is_80_10_10_and_deterministic() -> None:
    rows = _rows(1000)
    first = _split_rows(rows)
    second = _split_rows(rows)

    assert len(first["train"]) == 800
    assert len(first["validation"]) == 100
    assert len(first["test"]) == 100
    # Deterministic: same input -> identical splits.
    assert first["test"] == second["test"]


def test_assert_healthy_split_passes_on_full_corpus() -> None:
    # Should not raise for a sane 80/10/10 split.
    _assert_healthy_split("aka_asr", _split_rows(_rows(1000)))


def test_assert_healthy_split_raises_on_truncated_test_split() -> None:
    # Mimic the stale-download artifact: large train, 1-row test.
    truncated = {
        "train": _rows(10107),
        "validation": _rows(1123),
        "test": _rows(1),
    }
    with pytest.raises(ValueError, match="truncated"):
        _assert_healthy_split("aka_asr", truncated)


def test_assert_healthy_split_skips_small_corpora() -> None:
    # Below the 100-row floor (e.g. an intentional --asr-limit run) the guard is a no-op.
    _assert_healthy_split("aka_asr", _split_rows(_rows(20)))
