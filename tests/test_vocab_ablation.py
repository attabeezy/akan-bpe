from __future__ import annotations

from pathlib import Path

import numpy as np

from akan_bpe.vocab_ablation import (
    embedding_costs,
    paired_bootstrap_fertility_difference,
    render_fertility_svg,
    save_fertility_figure,
    select_operating_point,
)


def test_paired_bootstrap_reports_negative_candidate_gain() -> None:
    result = paired_bootstrap_fertility_difference(
        candidate_token_counts=np.asarray([2, 4, 3]),
        baseline_token_counts=np.asarray([4, 8, 6]),
        word_counts=np.asarray([2, 2, 3]),
        resamples=200,
        seed=42,
    )

    assert result["difference_candidate_minus_baseline"] < 0
    assert result["upper"] < 0
    assert result["resamples"] == 200


def test_operating_point_chooses_smallest_vocab_on_both_domain_plateaus() -> None:
    rows = [
        {"vocab_size": 4000, "asr_fertility": 1.20, "formal_fertility": 1.25},
        {"vocab_size": 8000, "asr_fertility": 1.19, "formal_fertility": 1.24},
        {"vocab_size": 16000, "asr_fertility": 1.185, "formal_fertility": 1.235},
        {"vocab_size": 32000, "asr_fertility": 1.18, "formal_fertility": 1.23},
    ]

    result = select_operating_point(rows, relative_plateau_tolerance=0.01)

    assert result["selected_vocab_size"] == 8000
    assert result["qualifying_vocab_sizes"] == [8000, 16000, 32000]
    assert result["status"] == "plateau_observed"


def test_operating_point_marks_largest_only_selection_as_boundary() -> None:
    rows = [
        {"vocab_size": 4000, "asr_fertility": 1.38, "formal_fertility": 1.32},
        {"vocab_size": 8000, "asr_fertility": 1.30, "formal_fertility": 1.27},
        {"vocab_size": 16000, "asr_fertility": 1.24, "formal_fertility": 1.23},
        {"vocab_size": 32000, "asr_fertility": 1.20, "formal_fertility": 1.21},
    ]

    result = select_operating_point(rows, relative_plateau_tolerance=0.01)

    assert result["selected_vocab_size"] == 32000
    assert result["status"] == "boundary_selected_no_earlier_plateau"
    assert result["boundary_caveat"]


def test_embedding_cost_counts_untied_input_and_output_matrices() -> None:
    costs = embedding_costs(8000)

    assert costs["Qwen/Qwen3-0.6B-Base"]["input_plus_output_parameters"] == 16_384_000
    assert costs["Qwen/Qwen3-0.6B-Base"]["fp16_mib"] == 31.25


def test_svg_has_title_series_labels_and_zero_axis(tmp_path: Path) -> None:
    payload = {
        "tradeoff_rows": [
            {"vocab_size": 4000, "asr_fertility": 1.3, "formal_fertility": 1.25},
            {"vocab_size": 8000, "asr_fertility": 1.2, "formal_fertility": 1.2},
            {"vocab_size": 16000, "asr_fertility": 1.18, "formal_fertility": 1.19},
            {"vocab_size": 32000, "asr_fertility": 1.17, "formal_fertility": 1.18},
        ]
    }

    svg = render_fertility_svg(payload)
    output = tmp_path / "figure.svg"
    output.write_text(svg, encoding="utf-8")

    assert "Fertility by vocabulary size" in svg
    assert "ASR v2" in svg
    assert "Formal Twi" in svg
    assert ">0.0<" in svg
    assert output.stat().st_size > 1000

    rendered_svg = tmp_path / "rendered.svg"
    rendered_png = tmp_path / "rendered.png"
    save_fertility_figure(payload, svg_path=rendered_svg, png_path=rendered_png)
    assert rendered_svg.stat().st_size > 5000
    assert rendered_png.stat().st_size > 5000

    svg_bytes = rendered_svg.read_bytes()
    png_bytes = rendered_png.read_bytes()
    save_fertility_figure(payload, svg_path=rendered_svg, png_path=rendered_png)
    assert rendered_svg.read_bytes() == svg_bytes
    assert rendered_png.read_bytes() == png_bytes
