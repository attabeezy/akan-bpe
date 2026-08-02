from __future__ import annotations

import json
from pathlib import Path

from akan_bpe.revision_comparison import (
    AKAN_TOKENIZERS,
    BASELINE_TOKENIZERS,
    TEST_SETS,
    build_revision_comparison,
    render_revision_comparison_markdown,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fertility_payload(asr_shift: float = 0.0, asr_path: str = "v1.jsonl") -> dict[str, object]:
    values = {
        "xlm_roberta_base": (2.4, 2.5),
        "bert_base_multilingual_cased": (2.3, 2.4),
        "mt5_base": (2.35, 2.55),
        "asr": (1.2, 1.5),
        "tts": (1.45, 1.25),
        "mixed": (1.3, 1.3),
    }
    return {
        "test_sets": {"asr_test": asr_path, "tts_test": "formal.jsonl"},
        "results": {
            name: {
                "asr_test": {"fertility": asr + asr_shift},
                "tts_test": {"fertility": formal},
            }
            for name, (asr, formal) in values.items()
        },
        "summary": {"best_on_asr_test": "asr", "best_on_tts_test": "tts"},
    }


def _router_payload(correct: int, total: int) -> dict[str, object]:
    return {
        "total_samples": total,
        "routing_decisions": {"asr": correct, "tts": total - correct, "mixed": 0},
    }


def test_revision_comparison_retains_conclusion_for_tiny_change(tmp_path: Path) -> None:
    paths = {
        "fertility_v1_path": tmp_path / "fertility-v1.json",
        "fertility_v2_path": tmp_path / "fertility-v2.json",
        "heuristic_router_v1_path": tmp_path / "heuristic-v1.json",
        "heuristic_router_v2_path": tmp_path / "heuristic-v2.json",
        "ml_router_v1_path": tmp_path / "ml-v1.json",
        "ml_router_v2_path": tmp_path / "ml-v2.json",
    }
    _write_json(paths["fertility_v1_path"], _fertility_payload())
    _write_json(paths["fertility_v2_path"], _fertility_payload(0.0001, "v2.jsonl"))
    _write_json(paths["heuristic_router_v1_path"], _router_payload(800, 1000))
    _write_json(paths["heuristic_router_v2_path"], _router_payload(799, 999))
    _write_json(paths["ml_router_v1_path"], _router_payload(1000, 1000))
    _write_json(paths["ml_router_v2_path"], _router_payload(999, 999))

    payload = build_revision_comparison(**paths)

    assert set(payload["fertility_changes"]) == set(  # type: ignore[arg-type]
        (*BASELINE_TOKENIZERS, *AKAN_TOKENIZERS)
    )
    assert all(  # type: ignore[union-attr]
        set(test_sets) == set(TEST_SETS)
        for test_sets in payload["fertility_changes"].values()  # type: ignore[union-attr]
    )
    assert payload["decision"]["material_change"] is False  # type: ignore[index]
    assert payload["decision"]["model_ladder_disposition"] == (  # type: ignore[index]
        "retain_existing_model_ladder"
    )
    markdown = render_revision_comparison_markdown(payload)  # type: ignore[arg-type]
    assert "# ASR Split Revision" in markdown
    assert "Material change: **false**" in markdown


def test_revision_comparison_stops_on_material_change(tmp_path: Path) -> None:
    paths = {
        "fertility_v1_path": tmp_path / "fertility-v1.json",
        "fertility_v2_path": tmp_path / "fertility-v2.json",
        "heuristic_router_v1_path": tmp_path / "heuristic-v1.json",
        "heuristic_router_v2_path": tmp_path / "heuristic-v2.json",
        "ml_router_v1_path": tmp_path / "ml-v1.json",
        "ml_router_v2_path": tmp_path / "ml-v2.json",
    }
    _write_json(paths["fertility_v1_path"], _fertility_payload())
    _write_json(paths["fertility_v2_path"], _fertility_payload(0.1, "v2.jsonl"))
    for key in ("heuristic_router_v1_path", "heuristic_router_v2_path"):
        _write_json(paths[key], _router_payload(800, 1000))
    for key in ("ml_router_v1_path", "ml_router_v2_path"):
        _write_json(paths[key], _router_payload(1000, 1000))

    payload = build_revision_comparison(**paths)

    assert payload["decision"]["material_change"] is True  # type: ignore[index]
    assert payload["decision"]["model_ladder_disposition"] == (  # type: ignore[index]
        "stop_and_investigate_before_new_experiments"
    )
