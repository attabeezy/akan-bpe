from __future__ import annotations

from akan_bpe.router_audit import _metrics


def test_metrics_uses_explicit_label_order_and_confusion_orientation() -> None:
    result = _metrics([0, 0, 1, 1], [0, 1, 1, 1])

    assert result["accuracy"] == 0.75
    assert result["confusion_matrix"] == {
        "label_order": ["asr", "tts"],
        "rows_true_columns_predicted": [[1, 1], [0, 2]],
    }
    assert result["per_class"]["asr"]["recall"] == 0.5
    assert result["per_class"]["tts"]["precision"] == 2 / 3
