from __future__ import annotations

import json
import sys
from pathlib import Path

from akan_bpe.model_integration import (
    ModelIntegrationConfig,
    PeftConfigSpec,
    build_result_payload,
    build_text_dataset,
    compute_token_count_comparison,
    load_experiment_tokenizer,
)
from akan_bpe.tokenizers import train_bpe_tokenizer


def test_load_experiment_tokenizer_and_build_text_dataset(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tts_tokenizer.json"
    train_bpe_tokenizer(
        texts=["akwaaba ma me", "me din de kodwo"],
        output_path=tokenizer_path,
        vocab_size=64,
        name="tts",
    )

    tokenizer = load_experiment_tokenizer(tokenizer_path)
    dataset = build_text_dataset(["akwaaba ma me"], tokenizer, max_length=8)

    row = dataset[0]
    assert tokenizer.pad_token is not None
    assert len(row["input_ids"]) == 8
    # Padding positions must be masked with -100 so they are excluded from the loss.
    # Non-padding positions must match input_ids exactly.
    for label, mask, token_id in zip(row["labels"], row["attention_mask"], row["input_ids"]):
        if mask == 1:
            assert label == token_id
        else:
            assert label == -100


def test_compute_token_count_comparison_uses_base_and_experiment_tokenizers(
    tmp_path: Path, monkeypatch
) -> None:
    tokenizer_path = tmp_path / "tts_tokenizer.json"
    train_bpe_tokenizer(
        texts=["akwaaba ma me", "me din de kodwo"],
        output_path=tokenizer_path,
        vocab_size=64,
        name="tts",
    )
    experiment_tokenizer = load_experiment_tokenizer(tokenizer_path)

    class FakeBaseTokenizer:
        pad_token = None
        eos_token = "</s>"
        unk_token = "<unk>"

        def __call__(self, text: str, add_special_tokens: bool = False):
            return {"input_ids": list(range(len(text.split()) * 2))}

    monkeypatch.setattr(
        "akan_bpe.model_integration.AutoTokenizer.from_pretrained",
        lambda model_id: FakeBaseTokenizer(),
    )

    payload = compute_token_count_comparison(
        model_id="fake/model",
        experiment_tokenizer=experiment_tokenizer,
        texts=["akwaaba ma me"],
    )

    assert payload["base_model_tokenizer"]["total_tokens"] == 6
    assert payload["experiment_tokenizer"]["total_tokens"] > 0
    assert payload["token_reduction_ratio"] <= 1.0


def test_build_result_payload_contains_required_fields() -> None:
    config = ModelIntegrationConfig(
        experiment_id="exp001",
        model_id="Qwen/Qwen3-0.6B",
        tokenizer_path="models/tts_tokenizer.json",
        train_file="data/pristine_twi_train.jsonl",
        eval_file="data/pristine_twi_test.jsonl",
        output_dir="models/exp001",
        results_output="results/exp001.json",
        peft=PeftConfigSpec(),
    )
    payload = build_result_payload(
        config=config,
        train_texts=["a", "b"],
        eval_texts=["c"],
        token_count_comparison={"token_reduction_ratio": 0.25},
        eval_metrics={"eval_loss": 1.0, "perplexity": 2.0},
        generation_samples=[{"prompt": "akwaaba", "completion": "akwaaba me nua"}],
        device={"cuda_available": False, "device_name": "cpu", "device_count": 0},
    )

    assert payload["experiment_id"] == "exp001"
    assert payload["eval"]["perplexity"] == 2.0
    assert payload["generation_samples"]
    assert payload["peft"]["rank"] == 16


def test_model_integration_cli_writes_results_json(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "model_output"
    results_path = tmp_path / "result.json"

    from scripts import model_integration as cli

    def fake_run_model_integration(config: ModelIntegrationConfig) -> dict[str, object]:
        assert config.experiment_id == "exp_cli"
        return {
            "experiment_id": config.experiment_id,
            "model_id": config.model_id,
            "output_model_dir": config.output_dir,
            "eval": {"eval_loss": 1.0, "perplexity": 2.0},
            "generation_samples": [{"prompt": "akwaaba", "completion": "akwaaba"}],
            "token_count_comparison": {"token_reduction_ratio": 0.1},
        }

    monkeypatch.setattr(cli, "run_model_integration", fake_run_model_integration)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/model_integration.py",
            "--experiment-id",
            "exp_cli",
            "--model-id",
            "fake/model",
            "--tokenizer-path",
            "models/tts_tokenizer.json",
            "--train-file",
            "data/pristine_twi_train.jsonl",
            "--eval-file",
            "data/pristine_twi_test.jsonl",
            "--output-dir",
            str(output_dir),
            "--results-output",
            str(results_path),
        ],
    )

    cli.main()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "exp_cli"
    assert Path(payload["output_model_dir"]) == output_dir
