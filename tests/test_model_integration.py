from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from akan_bpe.model_integration import (
    DEFAULT_SMOKE_MODEL_ID,
    ModelIntegrationConfig,
    PeftConfigSpec,
    build_generation_eval_examples,
    build_result_payload,
    build_text_dataset,
    compute_generation_quality_metrics,
    compute_token_count_comparison,
    load_experiment_tokenizer,
    run_model_integration,
    validate_colab_qlora_config,
)
from akan_bpe.datasets import TextSample
from akan_bpe.tokenizers import train_bpe_tokenizer


def test_training_arguments_kwargs_are_valid() -> None:
    """Regression: notebook crashed with 'unexpected keyword argument evaluation_strategy'.
    Verify the kwargs _build_model_and_training_args passes to TrainingArguments are accepted."""
    import ast
    import inspect

    from transformers import TrainingArguments

    valid_params = set(inspect.signature(TrainingArguments).parameters)

    source = inspect.getsource(
        __import__(
            "akan_bpe.model_integration", fromlist=["_build_model_and_training_args"]
        )._build_model_and_training_args
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_training_args = (isinstance(func, ast.Name) and func.id == "training_arguments_cls") or (
            isinstance(func, ast.Attribute) and func.attr == "TrainingArguments"
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


def test_build_generation_eval_examples_splits_prompt_and_reference() -> None:
    samples = [
        TextSample(id="too_short", text="one two three", source="test"),
        TextSample(
            id="ok",
            text=" ".join(f"w{i}" for i in range(1, 11)),
            source="test",
        ),
    ]

    rows = build_generation_eval_examples(
        samples=samples,
        max_samples=1,
        prompt_words=4,
        reference_words=3,
    )

    assert rows == [
        {
            "id": "ok",
            "prompt": "w1 w2 w3 w4",
            "reference": "w5 w6 w7",
        }
    ]


def test_compute_generation_quality_metrics_preserves_reconstructable_rows() -> None:
    pytest.importorskip("sacrebleu")
    rows = [
        {
            "id": "sample1",
            "prompt": "me din",
            "reference": "de kwame",
            "hypothesis": "de kwame",
            "full_generation": "me din de kwame",
        }
    ]

    payload = compute_generation_quality_metrics(
        rows=rows,
        prompt_words=2,
        reference_words=2,
        max_new_tokens=4,
        batch_size=1,
    )

    assert payload["metric"] == "sacrebleu.CHRF"
    assert payload["num_examples"] == 1
    assert payload["chrf"] > 99.0
    assert payload["chrfpp"] > 99.0
    assert payload["examples"] == rows


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
        model_id="sshleifer/tiny-gpt2",
        tokenizer_path="models/tts_tokenizer.json",
        train_file="data/pristine_twi_train.jsonl",
        eval_file="data/pristine_twi_test.jsonl",
        output_dir="models/bad001",
        results_output="results/bad001.json",
        device_mode="colab-qlora",
    )

    with pytest.raises(ValueError, match="does not support this model"):
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
    monkeypatch.setattr(
        "akan_bpe.model_integration.load_experiment_tokenizer", lambda path: tokenizer
    )
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
    ) -> tuple[dict[str, object], list[dict[str, str]], dict[str, object], dict[str, object]]:
        called["smoke"] = True
        assert runtime_model_id == DEFAULT_SMOKE_MODEL_ID
        assert len(eval_dataset) == 2
        return (
            {"eval_loss": 1.0, "perplexity": 2.0, "bpb": {"experiment": {"bits_per_byte": 1.5}}},
            [{"prompt": "akwaaba", "completion": "akwaaba me nua"}],
            {
                "enabled": True,
                "validated_pipeline_only": True,
                "forward_pass_succeeded": True,
                "generation_succeeded": True,
                "embedding_resize_succeeded": True,
            },
            {"mode": "random", "rows_initialized": 0},
        )

    monkeypatch.setattr(
        "akan_bpe.model_integration._build_model_and_training_args",
        fake_build_model_and_training_args,
    )
    monkeypatch.setattr(
        "akan_bpe.model_integration._run_smoke_validation", fake_run_smoke_validation
    )

    payload = run_model_integration(config)

    assert called["smoke"] is True
    assert called["build_model"] is False
    assert payload["runtime_model_id"] == DEFAULT_SMOKE_MODEL_ID
    assert payload["output_model_dir"] is None
    assert payload["smoke"]["validated_pipeline_only"] is True
    assert payload["embedding_init_mode"] == "random"
    assert payload["embedding_init"] == {"mode": "random", "rows_initialized": 0}
    assert payload["eval"]["bpb"]["experiment"]["bits_per_byte"] == 1.5


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
    monkeypatch.setattr(
        "akan_bpe.model_integration.load_experiment_tokenizer", lambda path: tokenizer
    )
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
        class cuda:  # noqa: N801 - mirrors the torch.cuda namespace
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
        return (
            FakeModel(),
            object(),
            {"cuda_available": False, "device_count": 0, "device_name": "cpu"},
            {"mode": "random", "rows_initialized": 0},
        )

    monkeypatch.setattr(
        "akan_bpe.model_integration.compute_bpb_metrics",
        lambda **kwargs: {
            "experiment": {"bits_per_byte": 1.2},
            "base": {"bits_per_byte": 1.8},
            "improvement": 0.6,
            "total_bytes": 42,
        },
    )

    def fake_run_smoke_validation(*args, **kwargs):
        trainer_called["smoke"] = True
        raise AssertionError("QLoRA mode should not use smoke validation")

    monkeypatch.setattr(
        "akan_bpe.model_integration._build_model_and_training_args",
        fake_build_model_and_training_args,
    )
    monkeypatch.setattr(
        "akan_bpe.model_integration._run_smoke_validation", fake_run_smoke_validation
    )
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
    assert payload["eval"]["bpb"]["improvement"] == 0.6
    assert payload["embedding_init"] == {"mode": "random", "rows_initialized": 0}


class _FakeTokenizer:
    """Returns fixed input_ids regardless of text, for deterministic BPB tests."""

    def __init__(self, ids, eos_token_id=None):
        self._ids = list(ids)
        self.eos_token_id = eos_token_id

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(self._ids)}


class _FakeOutputs:
    def __init__(self, logits):
        self.logits = logits


class _FakeBpbModel:
    device = "cpu"

    def __init__(self, logits):
        self._logits = logits

    def eval(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None):
        return _FakeOutputs(self._logits)


def test_compute_model_bpb_full_sums_nll_over_all_targets() -> None:
    import torch

    from akan_bpe.model_integration import compute_model_bpb_full

    # One chunk of 2 tokens, vocab size 2, uniform logits -> the single target costs
    # -log(0.5) = ln(2) nats == 1 bit. The 10-byte text gives 1 bit / 10 bytes = 0.1 BPB.
    model = _FakeBpbModel(torch.zeros(1, 2, 2))
    tokenizer = _FakeTokenizer([0, 1])
    result = compute_model_bpb_full(model, ["x" * 10], tokenizer, torch, chunk_size=256)

    assert result.num_target_tokens == 1
    assert result.total_bytes == 10
    assert result.total_nll_bits == pytest.approx(1.0)
    assert result.bits_per_byte == pytest.approx(0.1)


def test_compute_model_bpb_full_covers_every_byte_no_truncation() -> None:
    """Regression for the truncation bug: the denominator is the FULL byte count,
    and every token of a long text is scored across multiple chunks."""
    import torch

    from akan_bpe.metrics import count_utf8_bytes
    from akan_bpe.model_integration import compute_model_bpb_full

    # 600 tokens > chunk_size 256 -> 3 chunks (256 + 256 + 88); targets = 255 + 255 + 87 = 597.
    ids = list(range(600))

    class _ChunkModel:
        device = "cpu"

        def eval(self):
            return self

        def __call__(self, input_ids=None, attention_mask=None):
            return _FakeOutputs(torch.zeros(1, input_ids.shape[1], 600))

    text = "y" * 1000
    result = compute_model_bpb_full(
        _ChunkModel(), [text], _FakeTokenizer(ids), torch, chunk_size=256
    )

    assert result.num_target_tokens == 597  # every token scored, none dropped
    assert result.total_bytes == count_utf8_bytes([text]) == 1000  # full coverage, not a prefix


def test_compute_model_bpb_full_skips_single_token_chunks() -> None:
    import torch

    from akan_bpe.model_integration import compute_model_bpb_full

    # A lone token has no target -> skipped, zero NLL.
    model = _FakeBpbModel(torch.zeros(1, 1, 2))
    result = compute_model_bpb_full(model, ["x" * 10], _FakeTokenizer([0]), torch, chunk_size=256)

    assert result.num_target_tokens == 0
    assert result.total_nll_bits == pytest.approx(0.0)
    assert result.bits_per_byte == pytest.approx(0.0)


def test_init_embeddings_mean_of_subword_averages_base_rows() -> None:
    import torch
    from torch import nn

    from akan_bpe.model_integration import _init_embeddings_mean_of_subword

    class FakeEmbModel:
        def __init__(self, vocab: int, dim: int) -> None:
            self._inp = nn.Embedding(vocab, dim)
            self._out = nn.Embedding(vocab, dim)

        def get_input_embeddings(self):
            return self._inp

        def get_output_embeddings(self):
            return self._out

    class FakeExpTokenizer:
        def __init__(self, surfaces: list[str]) -> None:
            self._surfaces = surfaces

        def __len__(self) -> int:
            return len(self._surfaces)

        def convert_ids_to_tokens(self, token_id: int) -> str:
            return self._surfaces[token_id]

        def convert_tokens_to_string(self, tokens: list[str]) -> str:
            return tokens[0]

    class FakeBaseTokenizer:
        def __init__(self, mapping: dict[str, list[int]]) -> None:
            self._mapping = mapping

        def encode(self, surface: str, add_special_tokens: bool = False) -> list[int]:
            return self._mapping.get(surface, [])

    base_embeddings = torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0]])  # global mean [3, 3]
    model = FakeEmbModel(vocab=2, dim=2)
    exp_tokenizer = FakeExpTokenizer(["a", "x"])
    base_tokenizer = FakeBaseTokenizer({"a": [0, 1], "x": []})  # "x" maps to nothing -> fallback

    rows = _init_embeddings_mean_of_subword(
        model=model,
        experiment_tokenizer=exp_tokenizer,
        base_tokenizer=base_tokenizer,
        base_input_embeddings=base_embeddings,
        base_output_embeddings=base_embeddings,
        torch=torch,
    )

    assert rows == 2
    # "a" -> mean of base rows 0 and 1 = [2, 2]; "x" -> global fallback mean [3, 3].
    assert torch.allclose(model.get_input_embeddings().weight[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(model.get_input_embeddings().weight[1], torch.tensor([3.0, 3.0]))


def test_resize_and_init_embeddings_random_is_noop(monkeypatch) -> None:
    import torch
    from torch import nn

    from akan_bpe.model_integration import resize_and_init_embeddings

    resized = {"called_with": None}

    class FakeModel:
        def __init__(self) -> None:
            self._inp = nn.Embedding(4, 2)
            self._out = nn.Embedding(4, 2)

        def get_input_embeddings(self):
            return self._inp

        def get_output_embeddings(self):
            return self._out

        def resize_token_embeddings(self, new_len, pad_to_multiple_of=None):
            resized["called_with"] = new_len

    class FakeTokenizer:
        def __len__(self) -> int:
            return 3

    # If random init ever touched AutoTokenizer this lambda would flip the flag.
    touched = {"auto_tokenizer": False}

    def fake_from_pretrained(model_id):
        touched["auto_tokenizer"] = True
        raise AssertionError("random init must not load the base tokenizer")

    monkeypatch.setattr(
        "akan_bpe.model_integration.AutoTokenizer.from_pretrained", fake_from_pretrained
    )

    config = ModelIntegrationConfig(
        experiment_id="e",
        model_id="m",
        tokenizer_path="t",
        train_file="tr",
        eval_file="ev",
        output_dir="o",
        results_output="r",
        embedding_init_mode="random",
    )
    record = resize_and_init_embeddings(config, FakeModel(), FakeTokenizer(), "m", torch)

    assert record == {"mode": "random", "rows_initialized": 0}
    assert resized["called_with"] == 3
    assert touched["auto_tokenizer"] is False


def test_build_result_payload_carries_bpb_and_embedding_init() -> None:
    config = ModelIntegrationConfig(
        experiment_id="exp",
        model_id="Qwen/Qwen3-0.6B",
        tokenizer_path="t",
        train_file="tr",
        eval_file="ev",
        output_dir="o",
        results_output="r",
        embedding_init_mode="mean_subword",
    )
    payload = build_result_payload(
        config=config,
        runtime_model_id="Qwen/Qwen3-0.6B",
        train_texts=["a"],
        eval_texts=["b"],
        token_count_comparison={"token_reduction_ratio": 0.1},
        eval_metrics={
            "eval_loss": 1.0,
            "perplexity": 2.0,
            "bpb": {"base": {"bits_per_byte": 1.8}, "experiment": {"bits_per_byte": 1.2}},
        },
        generation_samples=[],
        device={"cuda_available": False, "device_name": "cpu", "device_count": 0},
        output_model_dir="o",
        embedding_init={"mode": "mean_subword", "rows_initialized": 8000},
    )

    assert payload["embedding_init_mode"] == "mean_subword"
    assert payload["embedding_init"]["rows_initialized"] == 8000
    assert payload["eval"]["bpb"]["experiment"]["bits_per_byte"] == 1.2


def test_cli_parses_embedding_init_and_skip_base_bpb(monkeypatch) -> None:
    from scripts import model_integration as cli

    captured = {}

    def fake_run(config: ModelIntegrationConfig) -> dict[str, object]:
        captured["config"] = config
        return {"experiment_id": config.experiment_id, "output_model_dir": config.output_dir}

    monkeypatch.setattr(cli, "run_model_integration", fake_run)
    monkeypatch.setattr(cli, "write_json", lambda path, payload: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/model_integration.py",
            "--experiment-id",
            "exp",
            "--model-id",
            "fake/model",
            "--tokenizer-path",
            "t.json",
            "--train-file",
            "tr.jsonl",
            "--eval-file",
            "ev.jsonl",
            "--output-dir",
            "out",
            "--embedding-init-mode",
            "mean_subword",
            "--skip-base-bpb",
            "--generation-eval-samples",
            "512",
            "--generation-prompt-words",
            "48",
            "--generation-reference-words",
            "64",
            "--generation-eval-max-new-tokens",
            "64",
            "--generation-eval-batch-size",
            "8",
        ],
    )

    cli.main()

    assert captured["config"].embedding_init_mode == "mean_subword"
    assert captured["config"].compute_base_bpb is False
    assert captured["config"].generation_eval_samples == 512
    assert captured["config"].generation_prompt_words == 48
    assert captured["config"].generation_reference_words == 64
    assert captured["config"].generation_eval_max_new_tokens == 64
    assert captured["config"].generation_eval_batch_size == 8


def test_cli_skip_generation_quality_eval_overrides_sample_count(monkeypatch) -> None:
    from scripts import model_integration as cli

    captured = {}

    def fake_run(config: ModelIntegrationConfig) -> dict[str, object]:
        captured["config"] = config
        return {"experiment_id": config.experiment_id, "output_model_dir": config.output_dir}

    monkeypatch.setattr(cli, "run_model_integration", fake_run)
    monkeypatch.setattr(cli, "write_json", lambda path, payload: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/model_integration.py",
            "--experiment-id",
            "exp",
            "--model-id",
            "fake/model",
            "--tokenizer-path",
            "t.json",
            "--train-file",
            "tr.jsonl",
            "--eval-file",
            "ev.jsonl",
            "--output-dir",
            "out",
            "--generation-eval-samples",
            "512",
            "--skip-generation-quality-eval",
        ],
    )

    cli.main()

    assert captured["config"].generation_eval_samples == 0


def test_derive_experiment_id_uses_model_slug() -> None:
    from akan_bpe.model_integration import derive_experiment_id, model_slug

    assert model_slug("Qwen/Qwen3-0.6B-Base") == "qwen-0.6b"
    assert derive_experiment_id("Qwen/Qwen3-0.6B-Base", "random") == "run-qwen-0.6b-mixed"
    assert (
        derive_experiment_id("meta-llama/Llama-3.2-1B", "mean_subword")
        == "run-llama-1b-mixed-meansub"
    )


def test_cli_derives_defaults_from_model_id(monkeypatch) -> None:
    from scripts import model_integration as cli

    captured = {}

    monkeypatch.setattr(
        cli, "run_model_integration", lambda config: captured.update(config=config) or {}
    )
    monkeypatch.setattr(cli, "write_json", lambda path, payload: None)
    monkeypatch.setattr(
        sys, "argv", ["scripts/model_integration.py", "--model-id", "Qwen/Qwen3-0.6B-Base"]
    )

    cli.main()

    config = captured["config"]
    assert config.device_mode == "colab-qlora"  # real run is the default
    assert config.experiment_id == "run-qwen-0.6b-mixed"
    assert config.output_dir == str(Path("models") / "run-qwen-0.6b-mixed")
    assert config.results_output == str(Path("results") / "run-qwen-0.6b-mixed.json")
    assert config.tokenizer_path == "models/mixed_tokenizer.json"


def test_cli_requires_model_id_for_colab_qlora(monkeypatch) -> None:
    from scripts import model_integration as cli

    monkeypatch.setattr(sys, "argv", ["scripts/model_integration.py"])

    with pytest.raises(SystemExit):
        cli.main()
