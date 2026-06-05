from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from akan_bpe.model_integration import (
    DEFAULT_SMOKE_MODEL_ID,
    ModelIntegrationConfig,
    PeftConfigSpec,
    build_result_payload,
    build_text_dataset,
    compute_token_count_comparison,
    load_experiment_tokenizer,
    run_model_integration,
    validate_colab_qlora_config,
)
from akan_bpe.tokenizers import train_bpe_tokenizer


def test_training_arguments_kwargs_are_valid() -> None:
    """Regression: notebook crashed with 'unexpected keyword argument evaluation_strategy'.
    Verify the kwargs _build_model_and_training_args passes to TrainingArguments are accepted."""
    import ast
    import inspect

    from transformers import TrainingArguments

    valid_params = set(inspect.signature(TrainingArguments).parameters)

    source = inspect.getsource(
        __import__("akan_bpe.model_integration", fromlist=["_build_model_and_training_args"])._build_model_and_training_args
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_training_args = (
            (isinstance(func, ast.Name) and func.id == "training_arguments_cls")
            or (isinstance(func, ast.Attribute) and func.attr == "TrainingArguments")
        )
        if not is_training_args:
            continue
        for kw in node.keywords:
            if kw.arg is not None:
                assert kw.arg in valid_params, (
                    f"TrainingArguments does not accept '{kw.arg}'. "
                    f"Did you mean 'eval_strategy' instead of 'evaluation_strategy'?"
                )


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
    active_length = sum(row["attention_mask"])
    assert row["input_ids"][active_length - 1] == tokenizer.eos_token_id
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
        runtime_model_id="Qwen/Qwen3-0.6B",
        train_texts=["a", "b"],
        eval_texts=["c"],
        token_count_comparison={"token_reduction_ratio": 0.25},
        eval_metrics={"eval_loss": 1.0, "perplexity": 2.0},
        generation_samples=[{"prompt": "akwaaba", "completion": "akwaaba me nua"}],
        device={"cuda_available": False, "device_name": "cpu", "device_count": 0},
        output_model_dir="models/exp001",
        smoke=None,
        reload_verification=None,
        training=None,
    )

    assert payload["experiment_id"] == "exp001"
    assert payload["runtime_model_id"] == "Qwen/Qwen3-0.6B"
    assert payload["eval"]["perplexity"] == 2.0
    assert payload["generation_samples"]
    assert payload["peft"]["rank"] == 16


def test_validate_colab_qlora_config_rejects_unsupported_model() -> None:
    config = ModelIntegrationConfig(
        experiment_id="bad001",
        model_id="meta-llama/Llama-3.2-1B",
        tokenizer_path="models/tts_tokenizer.json",
        train_file="data/pristine_twi_train.jsonl",
        eval_file="data/pristine_twi_test.jsonl",
        output_dir="models/bad001",
        results_output="results/bad001.json",
        device_mode="colab-qlora",
    )

    with pytest.raises(ValueError, match="supports only the Qwen 2A1 path"):
        validate_colab_qlora_config(config)


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


def test_run_model_integration_smoke_skips_training_and_artifacts(monkeypatch) -> None:
    config = ModelIntegrationConfig(
        experiment_id="smoke001",
        model_id="Qwen/Qwen3-0.6B",
        tokenizer_path="models/tts_tokenizer.json",
        train_file="data/pristine_twi_train.jsonl",
        eval_file="data/pristine_twi_test.jsonl",
        output_dir="models/smoke001",
        results_output="results/smoke001.json",
        device_mode="smoke",
    )

    monkeypatch.setattr(
        "akan_bpe.model_integration.load_texts",
        lambda path, max_samples=None: ["akwaaba ma me", "me din de kodwo"],
    )
    tokenizer = load_experiment_tokenizer(Path("models/tts_tokenizer.json"))
    monkeypatch.setattr("akan_bpe.model_integration.load_experiment_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(
        "akan_bpe.model_integration.compute_token_count_comparison",
        lambda model_id, experiment_tokenizer, texts: {
            "base_model_tokenizer": {"total_tokens": 8},
            "experiment_tokenizer": {"total_tokens": 6},
            "token_reduction_ratio": 0.25,
        },
    )

    called = {"build_model": False, "smoke": False}

    def fake_build_model_and_training_args(*args, **kwargs):
        called["build_model"] = True
        raise AssertionError("smoke mode should not build training args")

    def fake_run_smoke_validation(
        config: ModelIntegrationConfig,
        runtime_model_id: str,
        tokenizer,
        eval_dataset,
        eval_texts,
        device,
    ) -> tuple[dict[str, float], list[dict[str, str]], dict[str, object]]:
        called["smoke"] = True
        assert runtime_model_id == DEFAULT_SMOKE_MODEL_ID
        assert len(eval_dataset) == 2
        return (
            {"eval_loss": 1.0, "perplexity": 2.0},
            [{"prompt": "akwaaba", "completion": "akwaaba me nua"}],
            {
                "enabled": True,
                "validated_pipeline_only": True,
                "forward_pass_succeeded": True,
                "generation_succeeded": True,
                "embedding_resize_succeeded": True,
            },
        )

    monkeypatch.setattr(
        "akan_bpe.model_integration._build_model_and_training_args",
        fake_build_model_and_training_args,
    )
    monkeypatch.setattr("akan_bpe.model_integration._run_smoke_validation", fake_run_smoke_validation)

    payload = run_model_integration(config)

    assert called["smoke"] is True
    assert called["build_model"] is False
    assert payload["runtime_model_id"] == DEFAULT_SMOKE_MODEL_ID
    assert payload["output_model_dir"] is None
    assert payload["smoke"]["validated_pipeline_only"] is True


def test_run_model_integration_colab_qlora_uses_training_branch(monkeypatch) -> None:
    config = ModelIntegrationConfig(
        experiment_id="train001",
        model_id="Qwen/Qwen3-0.6B",
        tokenizer_path="models/tts_tokenizer.json",
        train_file="data/pristine_twi_train.jsonl",
        eval_file="data/pristine_twi_test.jsonl",
        output_dir="models/train001",
        results_output="results/train001.json",
        device_mode="colab-qlora",
    )

    monkeypatch.setattr(
        "akan_bpe.model_integration.load_texts",
        lambda path, max_samples=None: ["akwaaba ma me", "me din de kodwo"],
    )
    class FakeTensor:
        def __init__(self, data):
            self.data = data

        def to(self, device):
            return self

        def item(self):
            return 2.718281828

    class FakeTokenizer:
        pad_token_id = 7
        eos_token_id = 6

        def save_pretrained(self, path: str) -> None:
            return None

        def decode(self, ids, skip_special_tokens: bool = True) -> str:
            return "akwaaba"

        def __call__(
            self,
            text: str,
            return_tensors=None,
            add_special_tokens=None,
            truncation=None,
            max_length=None,
            padding=None,
        ):
            if return_tensors == "pt":
                return {
                    "input_ids": FakeTensor([[1, 2, 3]]),
                    "attention_mask": FakeTensor([[1, 1, 1]]),
                }
            return {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
            }

        def __len__(self) -> int:
            return 32

    tokenizer = FakeTokenizer()
    monkeypatch.setattr("akan_bpe.model_integration.load_experiment_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(
        "akan_bpe.model_integration.compute_token_count_comparison",
        lambda model_id, experiment_tokenizer, texts: {
            "base_model_tokenizer": {"total_tokens": 8},
            "experiment_tokenizer": {"total_tokens": 6},
            "token_reduction_ratio": 0.25,
        },
    )

    class FakeTrainer:
        def __init__(self, *args, **kwargs) -> None:
            self.saved_path = None

        def train(self) -> None:
            return None

        def evaluate(self) -> dict[str, float]:
            return {"eval_loss": 1.0}

        def save_model(self, path: str) -> None:
            self.saved_path = path

    class FakeModel:
        def __init__(self) -> None:
            self.device = "cpu"

        def eval(self) -> None:
            return None

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def device_count() -> int:
                return 0

            @staticmethod
            def get_device_name(index: int) -> str:
                return "cpu"

        @staticmethod
        def tensor(value):
            class FakeTensor:
                def __init__(self, data):
                    self.data = data

                def item(self):
                    return 2.718281828

            return FakeTensor(value)

        @staticmethod
        def exp(value):
            return value

    trainer_called = {"build": False, "smoke": False}

    monkeypatch.setattr(
        "akan_bpe.model_integration._import_training_stack",
        lambda: {
            "Trainer": FakeTrainer,
            "default_data_collator": object(),
            "set_seed": lambda seed: None,
            "torch": FakeTorch,
        },
    )

    def fake_build_model_and_training_args(config, tokenizer, runtime_model_id):
        trainer_called["build"] = True
        assert runtime_model_id == "Qwen/Qwen3-0.6B"
        return FakeModel(), object(), {"cuda_available": False, "device_count": 0, "device_name": "cpu"}

    def fake_run_smoke_validation(*args, **kwargs):
        trainer_called["smoke"] = True
        raise AssertionError("QLoRA mode should not use smoke validation")

    monkeypatch.setattr(
        "akan_bpe.model_integration._build_model_and_training_args",
        fake_build_model_and_training_args,
    )
    monkeypatch.setattr("akan_bpe.model_integration._run_smoke_validation", fake_run_smoke_validation)
    monkeypatch.setattr(
        "akan_bpe.model_integration.verify_saved_qwen_artifacts",
        lambda config, runtime_model_id, prompt: {
            "success": True,
            "output_dir": config.output_dir,
            "runtime_model_id": runtime_model_id,
            "prompt": prompt,
            "completion": "akwaaba",
        },
    )
    payload = run_model_integration(config)

    assert trainer_called["build"] is True
    assert trainer_called["smoke"] is False
    assert payload["runtime_model_id"] == "Qwen/Qwen3-0.6B"
    assert payload["output_model_dir"] == "models/train001"
    assert payload["reload_verification"]["success"] is True
    assert payload["training"]["completed"] is True
