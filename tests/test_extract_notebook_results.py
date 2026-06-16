from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.extract_notebook_results import (
    NotebookResultError,
    build_notebook_results,
    extract_notebook_payload,
)


def _payload(split: str, model: str) -> dict:
    return {
        "split": split,
        "model_slugs": [model],
        "summary": {
            model: {
                "random": {
                    "eval_loss": 4.0,
                    "perplexity": 55.0,
                    "experiment_bpb": 1.5,
                    "base_bpb": 2.5,
                    "bpb_improvement": 1.0,
                    "base_fertility": 2.0,
                    "akan_fertility": 1.0,
                    "token_reduction_ratio": 0.5,
                    "generation_quality": {
                        "chrf": 12.0,
                        "chrfpp": 12.5,
                        "num_examples": 512,
                        "prompt_words": 48,
                        "reference_words": 64,
                        "max_new_tokens": 64,
                    },
                },
                "mean_subword": {
                    "eval_loss": 3.5,
                    "perplexity": 33.0,
                    "experiment_bpb": 1.2,
                    "base_bpb": 2.5,
                    "bpb_improvement": 1.3,
                    "base_fertility": 2.0,
                    "akan_fertility": 1.0,
                    "token_reduction_ratio": 0.5,
                    "generation_quality": {
                        "chrf": 14.0,
                        "chrfpp": 14.5,
                        "num_examples": 512,
                        "prompt_words": 48,
                        "reference_words": 64,
                        "max_new_tokens": 64,
                    },
                },
            }
        },
        "runs": {
            model: {
                "random": {
                    "experiment_id": f"run-{model}",
                    "model_id": "dummy/base",
                    "embedding_init_mode": "random",
                },
                "mean_subword": {
                    "experiment_id": f"run-{model}-meansub",
                    "model_id": "dummy/base",
                    "embedding_init_mode": "mean_subword",
                },
            }
        },
    }


def _write_notebook(path: Path, text: str) -> None:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [],
                "outputs": [{"output_type": "stream", "name": "stdout", "text": text}],
            }
        ],
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


def test_extract_notebook_payload_reads_full_json_block(tmp_path: Path) -> None:
    payload = _payload("light", "qwen-0.6b")
    notebook_path = tmp_path / "run-full-light.ipynb"
    _write_notebook(
        notebook_path,
        "noise\nBEGIN_NOTEBOOK_FULL_JSON light\n"
        + json.dumps(payload)
        + "\nEND_NOTEBOOK_FULL_JSON light\n",
    )

    extracted = extract_notebook_payload(notebook_path)

    assert extracted["split"] == "light"
    assert extracted["model_slugs"] == ["qwen-0.6b"]


def test_build_notebook_results_flattens_summary_and_interpretation(tmp_path: Path) -> None:
    light_path = tmp_path / "run-full-light.ipynb"
    heavy_path = tmp_path / "run-full-heavy.ipynb"
    _write_notebook(
        light_path,
        "BEGIN_NOTEBOOK_FULL_JSON light\n"
        + json.dumps(_payload("light", "qwen-0.6b"))
        + "\nEND_NOTEBOOK_FULL_JSON light\n",
    )
    _write_notebook(
        heavy_path,
        "BEGIN_NOTEBOOK_FULL_JSON heavy\n"
        + json.dumps(_payload("heavy", "gemma-1b"))
        + "\nEND_NOTEBOOK_FULL_JSON heavy\n",
    )

    result = build_notebook_results([light_path, heavy_path])

    assert result["source_of_truth"] == [str(light_path), str(heavy_path)]
    assert len(result["summary"]) == 4
    assert result["interpretation"]["mean_subword_chrf_wins"] == "2/2"
    assert result["interpretation"]["mean_subword_bpb_wins"] == "2/2"
    assert set(result["runs"]) == {"qwen-0.6b", "gemma-1b"}


def test_missing_result_marker_fails(tmp_path: Path) -> None:
    notebook_path = tmp_path / "missing.ipynb"
    _write_notebook(notebook_path, "no result here")

    with pytest.raises(NotebookResultError, match="No BEGIN_NOTEBOOK_FULL_JSON"):
        extract_notebook_payload(notebook_path)


def test_malformed_result_json_fails(tmp_path: Path) -> None:
    notebook_path = tmp_path / "malformed.ipynb"
    _write_notebook(
        notebook_path,
        "BEGIN_NOTEBOOK_FULL_JSON light\n{bad json}\nEND_NOTEBOOK_FULL_JSON light\n",
    )

    with pytest.raises(NotebookResultError, match="Malformed notebook result JSON"):
        extract_notebook_payload(notebook_path)


def test_duplicate_model_slug_fails(tmp_path: Path) -> None:
    first_path = tmp_path / "first.ipynb"
    second_path = tmp_path / "second.ipynb"
    payload = _payload("light", "qwen-0.6b")
    _write_notebook(
        first_path,
        "BEGIN_NOTEBOOK_FULL_JSON light\n"
        + json.dumps(payload)
        + "\nEND_NOTEBOOK_FULL_JSON light\n",
    )
    duplicate = _payload("heavy", "qwen-0.6b")
    _write_notebook(
        second_path,
        "BEGIN_NOTEBOOK_FULL_JSON heavy\n"
        + json.dumps(duplicate)
        + "\nEND_NOTEBOOK_FULL_JSON heavy\n",
    )

    with pytest.raises(NotebookResultError, match="Duplicate model slug"):
        build_notebook_results([first_path, second_path])
