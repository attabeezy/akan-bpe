#!/usr/bin/env python3
"""Staged LoRA training for dual-stream tokenization.

Implements all 6 experimental groups:
- Control: Standard Llama-3.2-1B (no fine-tuning)
- Variant A: ASR only
- Variant B: TTS only
- Variant C: ASR + TTS (mixed)
- Variant E: TTS -> ASR -> TTS (primary hypothesis)
- Variant G: ASR -> TTS

Usage:
    python scripts/02_train_lora.py --group E --data data/akan/ --output checkpoints/

Designed for Colab T4 GPU execution.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Literal
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType


TrainingGroup = Literal["control", "A", "B", "C", "E", "G"]


@dataclass
class TrainingConfig:
    """Configuration for a single training stage."""

    name: str
    data_split: Literal["asr", "tts", "mixed"]
    learning_rate: float
    epochs: int
    batch_size: int = 4
    max_length: int = 512


GROUP_CONFIGS: dict[TrainingGroup, list[TrainingConfig]] = {
    "control": [],
    "A": [TrainingConfig("asr_only", "asr", 2e-4, 3)],
    "B": [TrainingConfig("tts_only", "tts", 2e-4, 3)],
    "C": [TrainingConfig("mixed", "mixed", 2e-4, 3)],
    "E": [
        TrainingConfig("tts_stage1", "tts", 2e-4, 2),
        TrainingConfig("asr_stage2", "asr", 1e-4, 1),
        TrainingConfig("tts_stage3", "tts", 5e-5, 1),
    ],
    "G": [
        TrainingConfig("asr_stage1", "asr", 2e-4, 2),
        TrainingConfig("tts_stage2", "tts", 1e-4, 1),
    ],
}


def load_jsonl_texts(jsonl_path: Path) -> list[str]:
    """Load texts from JSONL file.

    Args:
        jsonl_path: Path to JSONL file with 'transcription' or 'text' field.

    Returns:
        List of text strings.
    """
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                text = item.get("transcription") or item.get("text", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def prepare_dataset(texts: list[str], tokenizer, max_length: int = 512) -> Dataset:
    """Prepare dataset for training.

    Args:
        texts: List of text samples.
        tokenizer: Tokenizer for encoding.
        max_length: Maximum sequence length.

    Returns:
        HuggingFace Dataset.
    """

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset


def train_stage(
    model, tokenizer, texts: list[str], config: TrainingConfig, output_dir: Path, stage_num: int
) -> None:
    """Execute a single training stage.

    Args:
        model: PEFT model to train.
        tokenizer: Tokenizer for encoding.
        texts: Training texts.
        config: Stage configuration.
        output_dir: Output directory for checkpoints.
        stage_num: Stage number for naming.
    """
    print(f"\n=== Stage {stage_num}: {config.name} ===")
    print(f"  Data split: {config.data_split}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Samples: {len(texts)}")

    dataset = prepare_dataset(texts, tokenizer, config.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / f"stage_{stage_num}_{config.name}"),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    print(f"Stage {stage_num} complete.")


def train_variant_e(
    model_id: str, data_dir: Path, output_dir: Path, configs: list[TrainingConfig]
) -> None:
    """Train Variant E: TTS -> ASR -> TTS staged training.

    Rationale: Anchor on formal logic, adapt to noise, refine logic.
    """
    print(f"\nLoading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    asr_file = data_dir / "aka_asr_train.jsonl"
    tts_file = data_dir / "twi_tts_train.jsonl"

    if not asr_file.exists():
        raise FileNotFoundError(f"ASR data not found: {asr_file}")
    if not tts_file.exists():
        raise FileNotFoundError(f"TTS data not found: {tts_file}")

    asr_texts = load_jsonl_texts(asr_file)
    tts_texts = load_jsonl_texts(tts_file)

    print(f"Loaded {len(asr_texts)} ASR samples")
    print(f"Loaded {len(tts_texts)} TTS samples")

    for stage_num, config in enumerate(configs, 1):
        if config.data_split == "asr":
            texts = asr_texts
        elif config.data_split == "tts":
            texts = tts_texts
        else:
            texts = asr_texts + tts_texts

        train_stage(model, tokenizer, texts, config, output_dir, stage_num)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model saved to: {final_dir}")


def train_variant_single(
    model_id: str, data_dir: Path, output_dir: Path, config: TrainingConfig
) -> None:
    """Train a single-stage variant (A, B, C)."""
    print(f"\nLoading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    asr_file = data_dir / "aka_asr_train.jsonl"
    tts_file = data_dir / "twi_tts_train.jsonl"

    texts = []
    if config.data_split in ["asr", "mixed"]:
        if asr_file.exists():
            texts.extend(load_jsonl_texts(asr_file))
    if config.data_split in ["tts", "mixed"]:
        if tts_file.exists():
            texts.extend(load_jsonl_texts(tts_file))

    if not texts:
        raise ValueError(f"No training data found for {config.data_split}")

    train_stage(model, tokenizer, texts, config, output_dir, 1)

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model saved to: {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA variants")
    parser.add_argument(
        "--group",
        type=str,
        default="E",
        choices=["control", "A", "B", "C", "E", "G"],
        help="Training group to execute",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--data", type=str, default="data/akan/")
    parser.add_argument("--output", type=str, default="checkpoints/")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per stage")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    output_dir = Path(args.output) / f"variant_{args.group}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    configs = GROUP_CONFIGS[args.group]

    if args.epochs is not None:
        for c in configs:
            c.epochs = args.epochs
    if args.lr is not None:
        for c in configs:
            c.learning_rate = args.lr

    print(f"Training group: {args.group}")
    print(f"Base model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Stages: {len(configs) if configs else 'None (control)'}")

    if args.group == "control":
        print("Control group: No training required. Use base model directly.")
        return

    if args.group == "E":
        train_variant_e(args.model, data_dir, output_dir, configs)
    else:
        train_variant_single(args.model, data_dir, output_dir, configs[0])

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
