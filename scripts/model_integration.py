#!/usr/bin/env python3
"""Run one Akan-BPE model-integration experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.io import write_json
from akan_bpe.model_integration import ModelIntegrationConfig, PeftConfigSpec, run_model_integration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Akan-BPE model-integration experiment.")
    parser.add_argument("--experiment-id", required=True, help="Stable identifier for this run.")
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model identifier. `colab-qlora` currently supports only Qwen/Qwen3-0.6B.",
    )
    parser.add_argument("--tokenizer-path", required=True, help="Local tokenizer JSON path.")
    parser.add_argument("--train-file", required=True, help="Training JSONL file.")
    parser.add_argument("--eval-file", required=True, help="Evaluation JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Model/adapters output directory.")
    parser.add_argument(
        "--results-output",
        help="Optional JSON output path. Defaults to results/<experiment-id>.json.",
    )
    parser.add_argument(
        "--device-mode",
        choices=("smoke", "colab-qlora"),
        default="smoke",
        help="Execution mode for tiny-model pipeline validation or the Qwen/Qwen3-0.6B Colab QLoRA path.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train cap.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional eval cap.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer sequence length.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target modules for LoRA.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--generation-samples",
        type=int,
        default=3,
        help="Number of eval prompts to generate for qualitative samples.",
    )
    parser.add_argument(
        "--generation-max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens per generation sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_output = args.results_output or str(Path("results") / f"{args.experiment_id}.json")
    peft = PeftConfigSpec(
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=tuple(
            module.strip() for module in args.lora_target_modules.split(",") if module.strip()
        ),
    )
    config = ModelIntegrationConfig(
        experiment_id=args.experiment_id,
        model_id=args.model_id,
        tokenizer_path=args.tokenizer_path,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        results_output=results_output,
        device_mode=args.device_mode,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        peft=peft,
        seed=args.seed,
        generation_samples=args.generation_samples,
        generation_max_new_tokens=args.generation_max_new_tokens,
    )

    payload = run_model_integration(config)
    write_json(Path(results_output), payload)
    print(f"Model integration results written to {results_output}")
    print(f"Model artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
