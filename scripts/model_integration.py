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
from akan_bpe.model_integration import (
    DEFAULT_EVAL_FILE,
    DEFAULT_SMOKE_MODEL_ID,
    DEFAULT_TOKENIZER_PATH,
    DEFAULT_TRAIN_FILE,
    ModelIntegrationConfig,
    PeftConfigSpec,
    derive_experiment_id,
    run_model_integration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Akan-BPE model-integration experiment.")
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model identifier (e.g. Qwen/Qwen3-0.6B-Base). Required for "
        "--device-mode colab-qlora; defaults to the tiny smoke model otherwise.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Stable run identifier. Defaults to a tag derived from the model id and "
        "embedding-init mode, e.g. run-qwen-0.6b / run-qwen-0.6b-meansub.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=DEFAULT_TOKENIZER_PATH,
        help=f"Local tokenizer JSON path. Defaults to {DEFAULT_TOKENIZER_PATH}.",
    )
    parser.add_argument(
        "--train-file",
        default=DEFAULT_TRAIN_FILE,
        help=f"Training JSONL file. Defaults to {DEFAULT_TRAIN_FILE}.",
    )
    parser.add_argument(
        "--eval-file",
        default=DEFAULT_EVAL_FILE,
        help=f"Evaluation JSONL file. Defaults to {DEFAULT_EVAL_FILE}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Model/adapters output directory. Defaults to models/<experiment-id>.",
    )
    parser.add_argument(
        "--results-output",
        default=None,
        help="JSON output path. Defaults to results/<experiment-id>.json.",
    )
    parser.add_argument(
        "--device-mode",
        choices=("smoke", "colab-qlora"),
        default="colab-qlora",
        help="Execution mode: 'colab-qlora' runs a real QLoRA fine-tune (default); "
        "'smoke' validates the pipeline on a tiny CPU model.",
    )
    parser.add_argument("--max-train-samples", type=int, default=4096, help="Train cap.")
    parser.add_argument("--max-eval-samples", type=int, default=512, help="Eval cap.")
    parser.add_argument("--max-length", type=int, default=256, help="Tokenizer sequence length.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps.")
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
    parser.add_argument(
        "--embedding-init-mode",
        choices=("random", "mean_subword"),
        default="random",
        help="Init for swapped-in Akan embeddings: 'random' (resize default) or "
        "'mean_subword' (mean of base subword embeddings).",
    )
    parser.add_argument(
        "--skip-base-bpb",
        dest="compute_base_bpb",
        action="store_false",
        help="Skip the extra base-model bits-per-byte pass (saves a second model load).",
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

    model_id = args.model_id
    if model_id is None:
        if args.device_mode == "smoke":
            model_id = DEFAULT_SMOKE_MODEL_ID
        else:
            raise SystemExit("--model-id is required for --device-mode colab-qlora.")

    experiment_id = args.experiment_id or derive_experiment_id(model_id, args.embedding_init_mode)
    output_dir = args.output_dir or str(Path("models") / experiment_id)
    results_output = args.results_output or str(Path("results") / f"{experiment_id}.json")

    peft = PeftConfigSpec(
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=tuple(
            module.strip() for module in args.lora_target_modules.split(",") if module.strip()
        ),
    )
    config = ModelIntegrationConfig(
        experiment_id=experiment_id,
        model_id=model_id,
        tokenizer_path=args.tokenizer_path,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=output_dir,
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
        embedding_init_mode=args.embedding_init_mode,
        compute_base_bpb=args.compute_base_bpb,
    )

    payload = run_model_integration(config)
    write_json(Path(results_output), payload)
    print(f"Model integration results written to {results_output}")
    print(f"Model artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
