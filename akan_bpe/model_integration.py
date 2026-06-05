"""Helpers for Phase 2 model-integration experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from akan_bpe.datasets import load_jsonl_samples, samples_to_texts
from akan_bpe.io import ensure_parent_dir

DEFAULT_SMOKE_MODEL_ID = "sshleifer/tiny-gpt2"
SUPPORTED_COLAB_QLORA_MODEL_IDS = ("Qwen/Qwen3-0.6B",)


@dataclass(frozen=True)
class PeftConfigSpec:
    """Serializable PEFT configuration for one experiment run."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Attention + MLP projections for Qwen3 (and most LLaMA-style models).
    # Including MLP layers (gate/up/down) alongside attention gives meaningfully
    # better QLoRA results; attention-only is a common under-powered default.
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class ModelIntegrationConfig:
    """Runtime configuration for one model-integration experiment."""

    experiment_id: str
    model_id: str
    tokenizer_path: str
    train_file: str
    eval_file: str
    output_dir: str
    results_output: str
    device_mode: str = "smoke"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_length: int = 256
    batch_size: int = 1
    grad_accum: int = 1
    epochs: float = 1.0
    learning_rate: float = 2e-4
    peft: PeftConfigSpec = field(default_factory=PeftConfigSpec)
    seed: int = 42
    generation_samples: int = 3
    generation_max_new_tokens: int = 32


def _set_model_token_config(model: Any, tokenizer: PreTrainedTokenizerFast) -> None:
    """Align model/generation token IDs with the experiment tokenizer."""
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is not None and hasattr(model.config, "bos_token_id"):
        model.config.bos_token_id = tokenizer.bos_token_id
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        if tokenizer.pad_token_id is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            generation_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None and hasattr(generation_config, "bos_token_id"):
            generation_config.bos_token_id = tokenizer.bos_token_id


def resolve_runtime_model_id(config: ModelIntegrationConfig) -> str:
    """Resolve the concrete model identifier used for a run."""
    if config.device_mode == "smoke":
        return DEFAULT_SMOKE_MODEL_ID
    return config.model_id


def load_texts(path: Path, max_samples: int | None = None) -> list[str]:
    """Load normalized texts from a JSONL file."""
    texts = samples_to_texts(load_jsonl_samples(path))
    if max_samples is None:
        return texts
    return texts[:max_samples]


def load_experiment_tokenizer(tokenizer_path: Path) -> PreTrainedTokenizerFast:
    """Load the local fast tokenizer used for model integration."""
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="[UNK]",
    )
    # Ensure pad_token is set even if not in the JSON, defaulting to a common fallback if needed.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    return tokenizer


def validate_colab_qlora_config(config: ModelIntegrationConfig) -> None:
    """Validate the Qwen-only Colab QLoRA contract before loading heavy dependencies."""
    if config.device_mode != "colab-qlora":
        return
    if config.model_id not in SUPPORTED_COLAB_QLORA_MODEL_IDS:
        supported = ", ".join(SUPPORTED_COLAB_QLORA_MODEL_IDS)
        raise ValueError(
            "`colab-qlora` currently supports only the Qwen 2A1 path. "
            f"Received model_id={config.model_id!r}; supported values: {supported}."
        )


def _build_causal_example(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
) -> dict[str, list[int]]:
    """Build one fixed-width causal LM example with explicit EOS termination."""
    if tokenizer.eos_token_id is None:
        raise ValueError("Experiment tokenizer must define an EOS token for model integration.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Experiment tokenizer must define a PAD token for model integration.")
    if max_length < 2:
        raise ValueError("max_length must be at least 2 for causal LM examples.")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length - 1,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    input_ids.append(tokenizer.eos_token_id)
    attention_mask.append(1)

    pad_length = max_length - len(input_ids)
    if pad_length > 0:
        input_ids.extend([tokenizer.pad_token_id] * pad_length)
        attention_mask.extend([0] * pad_length)

    labels = [
        token_id if mask == 1 else -100
        for token_id, mask in zip(input_ids, attention_mask)
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_text_dataset(
    texts: list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
) -> Dataset:
    """Tokenize texts into fixed-width causal LM training examples."""
    return Dataset.from_list(
        [_build_causal_example(text, tokenizer, max_length) for text in texts]
    )


def compute_token_count_stats(tokenizer: Any, texts: list[str]) -> dict[str, float | int]:
    """Measure total tokens and fertility-style token/word counts for a text list."""
    total_tokens = 0
    total_words = 0
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        total_tokens += len(input_ids)
        total_words += len(text.split())

    fertility = 0.0 if total_words == 0 else total_tokens / total_words
    return {
        "num_texts": len(texts),
        "total_tokens": total_tokens,
        "total_words": total_words,
        "fertility": fertility,
    }


def compute_token_count_comparison(
    model_id: str,
    experiment_tokenizer: PreTrainedTokenizerFast,
    texts: list[str],
) -> dict[str, object]:
    """Compare token counts between the base model tokenizer and the Akan tokenizer."""
    base_tokenizer = AutoTokenizer.from_pretrained(model_id)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = (
            base_tokenizer.eos_token or base_tokenizer.unk_token or base_tokenizer.pad_token
        )
    base_stats = compute_token_count_stats(base_tokenizer, texts)
    experiment_stats = compute_token_count_stats(experiment_tokenizer, texts)
    total_base_tokens = int(base_stats["total_tokens"])
    total_experiment_tokens = int(experiment_stats["total_tokens"])
    reduction_ratio = 0.0
    if total_base_tokens:
        reduction_ratio = (total_base_tokens - total_experiment_tokens) / total_base_tokens
    return {
        "base_model_tokenizer": base_stats,
        "experiment_tokenizer": experiment_stats,
        "token_reduction_ratio": reduction_ratio,
    }


def select_generation_prompts(texts: list[str], limit: int) -> list[str]:
    """Pick short prompts from eval texts for qualitative generation samples."""
    prompts = []
    for text in texts[:limit]:
        prompt = " ".join(text.split()[: min(12, len(text.split()))]).strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def build_result_payload(
    config: ModelIntegrationConfig,
    runtime_model_id: str,
    train_texts: list[str],
    eval_texts: list[str],
    token_count_comparison: dict[str, object],
    eval_metrics: dict[str, float],
    generation_samples: list[dict[str, str]],
    device: dict[str, object],
    output_model_dir: str | None,
    smoke: dict[str, object] | None = None,
    reload_verification: dict[str, object] | None = None,
    training: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create the stable JSON artifact for one experiment run."""
    return {
        "experiment_id": config.experiment_id,
        "model_id": config.model_id,
        "runtime_model_id": runtime_model_id,
        "tokenizer_path": config.tokenizer_path,
        "train_file": config.train_file,
        "eval_file": config.eval_file,
        "train_samples": len(train_texts),
        "eval_samples": len(eval_texts),
        "max_length": config.max_length,
        "batch_size": config.batch_size,
        "grad_accum": config.grad_accum,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "peft": asdict(config.peft),
        "device_mode": config.device_mode,
        "device": device,
        "token_count_comparison": token_count_comparison,
        "eval": eval_metrics,
        "generation_samples": generation_samples,
        "output_model_dir": output_model_dir,
        "smoke": smoke,
        "reload_verification": reload_verification,
        "training": training,
    }


def _import_runtime_stack() -> dict[str, Any]:
    """Load dependencies shared by smoke and training paths."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "Model integration requires torch and transformers. "
            'Install them with `pip install -e ".[train]"` for full training support.'
        ) from exc

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "set_seed": set_seed,
    }


def _import_training_stack() -> dict[str, Any]:
    """Load optional training dependencies lazily."""
    stack = _import_runtime_stack()
    try:
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError as exc:
        raise ImportError(
            "QLoRA model integration requires optional training dependencies. "
            'Install them with `pip install -e ".[train]"` and add `bitsandbytes` for QLoRA.'
        ) from exc

    return {
        **stack,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "LoraConfig": LoraConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "default_data_collator": default_data_collator,
        "get_peft_model": get_peft_model,
        "PeftModel": PeftModel,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
    }


def _validate_target_modules(model: Any, target_modules: tuple[str, ...]) -> None:
    """Fail early if the requested LoRA target modules are absent."""
    available = {name.split(".")[-1] for name, _module in model.named_modules()}
    missing = [name for name in target_modules if name not in available]
    if missing:
        preview = ", ".join(sorted(available)[:20])
        raise ValueError(
            "Invalid LoRA target modules for this model: "
            f"{', '.join(missing)}. Available module suffixes include: {preview}."
        )


def _build_model_and_training_args(
    config: ModelIntegrationConfig,
    tokenizer: PreTrainedTokenizerFast,
    runtime_model_id: str,
) -> tuple[Any, Any, dict[str, object]]:
    stack = _import_training_stack()
    torch = stack["torch"]
    auto_model_for_causal_lm = stack["AutoModelForCausalLM"]
    bits_and_bytes_config = stack["BitsAndBytesConfig"]
    lora_config_cls = stack["LoraConfig"]
    training_arguments_cls = stack["TrainingArguments"]
    get_peft_model = stack["get_peft_model"]
    prepare_model_for_kbit_training = stack["prepare_model_for_kbit_training"]

    if config.device_mode == "colab-qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("`colab-qlora` mode requires CUDA.")
        quantization_config = bits_and_bytes_config(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # T4 (Turing) does not have native bfloat16 tensor cores; float16 is
            # the right compute dtype here.
            bnb_4bit_compute_dtype=torch.float16,
        )
        # Omit device_map="auto" for training — HF docs say it is inference-only
        # and can cause unexpected device placement during the forward pass.
        model = auto_model_for_causal_lm.from_pretrained(
            runtime_model_id,
            quantization_config=quantization_config,
            dtype="auto",
        )
        fp16 = True
    else:
        model = auto_model_for_causal_lm.from_pretrained(runtime_model_id)
        fp16 = False

    _set_model_token_config(model, tokenizer)

    # Disable weight tying before resize so embed_tokens and lm_head remain
    # independent after the vocab is expanded.  Qwen3 ships with
    # tie_word_embeddings=True; keeping it set silently breaks the tie and
    # produces a noisy warning when both weights are present in the checkpoint.
    model.config.tie_word_embeddings = False

    # Resize BEFORE prepare_model_for_kbit_training and get_peft_model.
    # Resizing after PEFT wrapping can cause embedding size mismatches.
    # pad_to_multiple_of=64 aligns the vocab size to a hardware-friendly boundary.
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    _validate_target_modules(model, config.peft.target_modules)

    if config.device_mode == "colab-qlora":
        # use_reentrant=False avoids a PyTorch deprecation warning with newer
        # versions of torch that changed the default reentrant behaviour.
        model = prepare_model_for_kbit_training(
            model,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    peft_config = lora_config_cls(
        r=config.peft.rank,
        lora_alpha=config.peft.alpha,
        lora_dropout=config.peft.dropout,
        target_modules=list(config.peft.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
        # Save the resized embedding and LM head alongside the LoRA adapter so
        # the expanded vocab is not lost when loading from the adapter checkpoint.
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, peft_config)

    output_dir = Path(config.output_dir)
    ensure_parent_dir(output_dir / "adapter_config.json")
    training_args = training_arguments_cls(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=10,
        gradient_checkpointing=config.device_mode == "colab-qlora",
        report_to=[],
        remove_unused_columns=False,
        seed=config.seed,
        fp16=fp16,
    )

    device = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    return model, training_args, device


def _generate_samples(
    model: Any,
    tokenizer: PreTrainedTokenizerFast,
    prompts: list[str],
    max_length: int,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    """Generate deterministic qualitative samples for a prompt list."""
    generation_samples = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion = tokenizer.decode(generated[0], skip_special_tokens=True)
        generation_samples.append({"prompt": prompt, "completion": completion})
    return generation_samples


def _run_smoke_validation(
    config: ModelIntegrationConfig,
    runtime_model_id: str,
    tokenizer: PreTrainedTokenizerFast,
    eval_dataset: Dataset,
    eval_texts: list[str],
    device: dict[str, object],
) -> tuple[dict[str, float], list[dict[str, str]], dict[str, object]]:
    stack = _import_runtime_stack()
    torch = stack["torch"]
    auto_model_for_causal_lm = stack["AutoModelForCausalLM"]

    model = auto_model_for_causal_lm.from_pretrained(runtime_model_id)
    _set_model_token_config(model, tokenizer)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    model.eval()

    row = eval_dataset[0]
    batch = {
        key: torch.tensor([row[key]])
        for key in ("input_ids", "attention_mask", "labels")
    }
    with torch.no_grad():
        outputs = model(**batch)
    eval_loss = float(outputs.loss.item())
    perplexity = float(torch.exp(torch.tensor(eval_loss)).item())

    prompts = select_generation_prompts(eval_texts, config.generation_samples)
    with torch.no_grad():
        generation_samples = _generate_samples(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_length=config.max_length,
            max_new_tokens=config.generation_max_new_tokens,
        )

    smoke = {
        "enabled": True,
        "validated_pipeline_only": True,
        "forward_pass_succeeded": True,
        "generation_succeeded": True,
        "embedding_resize_succeeded": True,
    }
    return {"eval_loss": eval_loss, "perplexity": perplexity}, generation_samples, smoke


def verify_saved_qwen_artifacts(
    config: ModelIntegrationConfig,
    runtime_model_id: str,
    prompt: str,
) -> dict[str, object]:
    """Reload the saved tokenizer + adapter stack and verify inference works."""
    stack = _import_training_stack()
    auto_model_for_causal_lm = stack["AutoModelForCausalLM"]
    bits_and_bytes_config = stack["BitsAndBytesConfig"]
    peft_model_cls = stack["PeftModel"]
    torch = stack["torch"]

    output_dir = Path(config.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Expected output_dir to exist after training: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = bits_and_bytes_config(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = auto_model_for_causal_lm.from_pretrained(
        runtime_model_id,
        quantization_config=quantization_config,
        dtype="auto",
    )
    base_model.config.tie_word_embeddings = False
    _set_model_token_config(base_model, tokenizer)
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    reloaded_model = peft_model_cls.from_pretrained(base_model, str(output_dir))
    _set_model_token_config(reloaded_model, tokenizer)
    reloaded_model.eval()

    samples = _generate_samples(
        model=reloaded_model,
        tokenizer=tokenizer,
        prompts=[prompt],
        max_length=config.max_length,
        max_new_tokens=config.generation_max_new_tokens,
    )
    return {
        "success": True,
        "output_dir": str(output_dir),
        "runtime_model_id": runtime_model_id,
        "prompt": prompt,
        "completion": samples[0]["completion"] if samples else "",
    }


def run_model_integration(config: ModelIntegrationConfig) -> dict[str, object]:
    """Run one end-to-end model-integration experiment and return its result payload."""
    validate_colab_qlora_config(config)
    runtime_stack = _import_runtime_stack()
    torch = runtime_stack["torch"]
    set_seed = runtime_stack["set_seed"]
    trainer_cls = None
    default_data_collator = None
    if config.device_mode != "smoke":
        training_stack = _import_training_stack()
        trainer_cls = training_stack["Trainer"]
        default_data_collator = training_stack["default_data_collator"]

    set_seed(config.seed)
    train_texts = load_texts(Path(config.train_file), config.max_train_samples)
    eval_texts = load_texts(Path(config.eval_file), config.max_eval_samples)
    if not train_texts:
        raise ValueError(f"No train texts loaded from {config.train_file}")
    if not eval_texts:
        raise ValueError(f"No eval texts loaded from {config.eval_file}")

    runtime_model_id = resolve_runtime_model_id(config)
    tokenizer = load_experiment_tokenizer(Path(config.tokenizer_path))
    train_dataset = build_text_dataset(train_texts, tokenizer, config.max_length)
    eval_dataset = build_text_dataset(eval_texts, tokenizer, config.max_length)
    token_count_comparison = compute_token_count_comparison(runtime_model_id, tokenizer, eval_texts)

    if config.device_mode == "smoke":
        device = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        eval_metrics, generation_samples, smoke = _run_smoke_validation(
            config=config,
            runtime_model_id=runtime_model_id,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            eval_texts=eval_texts,
            device=device,
        )
        return build_result_payload(
            config=config,
            runtime_model_id=runtime_model_id,
            train_texts=train_texts,
            eval_texts=eval_texts,
            token_count_comparison=token_count_comparison,
            eval_metrics=eval_metrics,
            generation_samples=generation_samples,
            device=device,
            output_model_dir=None,
            smoke=smoke,
            reload_verification=None,
            training=None,
        )

    model, training_args, device = _build_model_and_training_args(config, tokenizer, runtime_model_id)

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # replaces deprecated tokenizer= (transformers 4.46+)
        data_collator=default_data_collator,
    )
    trainer.train()
    metrics = trainer.evaluate()
    eval_loss = float(metrics["eval_loss"])
    perplexity = float(torch.exp(torch.tensor(eval_loss)).item())

    model.eval()
    prompts = select_generation_prompts(eval_texts, config.generation_samples)
    generation_samples = _generate_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=config.max_length,
        max_new_tokens=config.generation_max_new_tokens,
    )

    output_dir = Path(config.output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    reload_prompt = prompts[0] if prompts else eval_texts[0]
    reload_verification = verify_saved_qwen_artifacts(
        config=config,
        runtime_model_id=runtime_model_id,
        prompt=reload_prompt,
    )

    return build_result_payload(
        config=config,
        runtime_model_id=runtime_model_id,
        train_texts=train_texts,
        eval_texts=eval_texts,
        token_count_comparison=token_count_comparison,
        eval_metrics={
            "eval_loss": eval_loss,
            "perplexity": perplexity,
        },
        generation_samples=generation_samples,
        device=device,
        output_model_dir=config.output_dir,
        smoke=None,
        reload_verification=reload_verification,
        training={
            "completed": True,
            "device_mode": config.device_mode,
        },
    )
