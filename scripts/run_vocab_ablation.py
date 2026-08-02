#!/usr/bin/env python3
"""Train and evaluate the frozen 4K/8K/16K/32K balanced mixed-BPE ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.datasets import load_jsonl_samples, samples_to_texts
from akan_bpe.io import ensure_parent_dir, write_json
from akan_bpe.revision_manifest import sha256_file
from akan_bpe.tokenizers import load_tokenizer
from akan_bpe.vocab_ablation import (
    BASELINE_TOKENIZERS,
    build_balanced_training_texts,
    embedding_costs,
    evaluate_tokenizer,
    paired_bootstrap_fertility_difference,
    render_tradeoff_markdown,
    save_fertility_figure,
    select_operating_point,
    train_vocab_variants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[4000, 8000, 16000, 32000])
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    parser.add_argument("--models-dir", type=Path, default=Path("models/revision_v2"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/vocab_ablation_results.json"),
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("results/vocab_ablation_tradeoff.md"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/vocab_ablation_fertility.svg"),
    )
    parser.add_argument(
        "--output-figure-png",
        type=Path,
        default=Path("results/vocab_ablation_fertility.png"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vocab_sizes = sorted(set(args.vocab_sizes))
    if vocab_sizes != [4000, 8000, 16000, 32000]:
        raise SystemExit("The frozen revision protocol requires exactly 4K, 8K, 16K, and 32K.")

    train_paths = [Path("data/aka_asr_train.jsonl"), Path("data/pristine_twi_train.jsonl")]
    asr_test_path = Path("data/revision_v2/aka_asr_test.jsonl")
    formal_test_path = Path("data/pristine_twi_test.jsonl")
    training_texts, original_counts = build_balanced_training_texts(train_paths)
    tokenizer_artifacts = cast(
        list[dict[str, Any]],
        train_vocab_variants(
            training_texts=training_texts,
            vocab_sizes=vocab_sizes,
            output_dir=args.models_dir,
            input_paths=train_paths,
            original_counts=original_counts,
        ),
    )

    datasets: dict[str, dict[str, Any]] = {
        "asr": {
            "path": asr_test_path,
            "texts": samples_to_texts(load_jsonl_samples(asr_test_path)),
        },
        "formal": {
            "path": formal_test_path,
            "texts": samples_to_texts(load_jsonl_samples(formal_test_path)),
        },
    }
    tokenizers: dict[str, tuple[Any, str]] = {
        name: (load_tokenizer(reference), reference)
        for name, reference in BASELINE_TOKENIZERS.items()
    }
    for artifact in tokenizer_artifacts:
        name = f"v{artifact['target_vocab_size']}"
        reference = str(artifact["path"])
        tokenizers[name] = (load_tokenizer(reference), reference)

    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    paired_counts: dict[str, dict[str, tuple[Any, Any]]] = {}
    for tokenizer_name, (tokenizer, reference) in tokenizers.items():
        metrics[tokenizer_name] = {}
        paired_counts[tokenizer_name] = {}
        for regime, dataset in datasets.items():
            evaluated, token_counts, word_counts = evaluate_tokenizer(
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                tokenizer_reference=reference,
                texts=dataset["texts"],
                source_file=dataset["path"].as_posix(),
            )
            metrics[tokenizer_name][regime] = evaluated
            paired_counts[tokenizer_name][regime] = (token_counts, word_counts)

    bootstrap: dict[str, dict[str, dict[str, object]]] = {}
    for vocab_size in vocab_sizes:
        candidate_name = f"v{vocab_size}"
        bootstrap[candidate_name] = {}
        for baseline_name in BASELINE_TOKENIZERS:
            bootstrap[candidate_name][baseline_name] = {}
            for regime in datasets:
                candidate_counts, word_counts = paired_counts[candidate_name][regime]
                baseline_counts, baseline_words = paired_counts[baseline_name][regime]
                if not (word_counts == baseline_words).all():
                    raise RuntimeError("Paired tokenizer evaluations produced mismatched words.")
                bootstrap[candidate_name][baseline_name][regime] = (
                    paired_bootstrap_fertility_difference(
                        candidate_token_counts=candidate_counts,
                        baseline_token_counts=baseline_counts,
                        word_counts=word_counts,
                        resamples=args.bootstrap_resamples,
                        seed=args.bootstrap_seed,
                    )
                )

    percentage_reductions: dict[str, dict[str, dict[str, float]]] = {}
    for vocab_size in vocab_sizes:
        candidate_name = f"v{vocab_size}"
        percentage_reductions[candidate_name] = {}
        for baseline_name in BASELINE_TOKENIZERS:
            percentage_reductions[candidate_name][baseline_name] = {}
            for regime in datasets:
                candidate_fertility = float(metrics[candidate_name][regime]["fertility"])
                baseline_fertility = float(metrics[baseline_name][regime]["fertility"])
                percentage_reductions[candidate_name][baseline_name][regime] = (
                    (baseline_fertility - candidate_fertility) / baseline_fertility * 100
                )

    historical_8k_path = Path("models/mixed_tokenizer.json")
    historical_8k = load_tokenizer(str(historical_8k_path))
    current_8k = tokenizers["v8000"][0]
    artifacts_by_size = {
        int(artifact["target_vocab_size"]): artifact for artifact in tokenizer_artifacts
    }
    encoding_mismatches: dict[str, int] = {}
    for regime, dataset in datasets.items():
        encoding_mismatches[regime] = sum(
            historical_8k.encode(text) != current_8k.encode(text) for text in dataset["texts"]
        )
    eight_k_compatibility = {
        "historical_path": historical_8k_path.as_posix(),
        "historical_sha256": sha256_file(historical_8k_path),
        "ablation_path": artifacts_by_size[8000]["path"],
        "ablation_sha256": artifacts_by_size[8000]["sha256"],
        "vocabularies_equal": historical_8k.get_vocab() == current_8k.get_vocab(),
        "encoding_mismatches": encoding_mismatches,
        "exact_test_encoding_match": all(value == 0 for value in encoding_mismatches.values()),
        "interpretation": (
            "The newly trained 8K tokenizer is behaviorally identical to the historical "
            "mixed tokenizer on both frozen test regimes; serialization hashes differ."
        ),
    }

    tradeoff_rows: list[dict[str, Any]] = []
    for vocab_size in vocab_sizes:
        name = f"v{vocab_size}"
        artifact = artifacts_by_size[vocab_size]
        asr = metrics[name]["asr"]
        formal = metrics[name]["formal"]
        tradeoff_rows.append(
            {
                "vocab_size": vocab_size,
                "actual_vocab_size": artifact["actual_vocab_size"],
                "asr_fertility": asr["fertility"],
                "asr_median": asr["sequence_length"]["median"],  # type: ignore[index]
                "asr_p90": asr["sequence_length"]["p90"],  # type: ignore[index]
                "asr_p95": asr["sequence_length"]["p95"],  # type: ignore[index]
                "asr_utilization_percent": asr["vocabulary_utilization"]["percent"],  # type: ignore[index]
                "formal_fertility": formal["fertility"],
                "formal_median": formal["sequence_length"]["median"],  # type: ignore[index]
                "formal_p90": formal["sequence_length"]["p90"],  # type: ignore[index]
                "formal_p95": formal["sequence_length"]["p95"],  # type: ignore[index]
                "formal_utilization_percent": formal["vocabulary_utilization"]["percent"],  # type: ignore[index]
                "tokenizer_size_mib": int(artifact["bytes"]) / (1024**2),
                "embedding_costs": embedding_costs(int(artifact["actual_vocab_size"])),
            }
        )
    operating_point = select_operating_point(tradeoff_rows, relative_plateau_tolerance=0.01)
    selected = operating_point["selected_vocab_size"]
    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": "balanced_mixed_bpe_vocab_ablation_revision_v2",
        "protocol": {
            "vocab_sizes": vocab_sizes,
            "varied_factor": "target_vocab_size_only",
            "training_files": [path.as_posix() for path in train_paths],
            "training_file_sha256": {
                path.as_posix(): sha256_file(path) for path in train_paths
            },
            "test_files": {
                regime: {
                    "path": dataset["path"].as_posix(),
                    "rows": len(dataset["texts"]),
                    "sha256": sha256_file(dataset["path"]),
                }
                for regime, dataset in datasets.items()
            },
            "bootstrap": {
                "method": "paired row resampling of aggregate fertility difference",
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "confidence_level": 0.95,
            },
        },
        "tokenizer_artifacts": tokenizer_artifacts,
        "metrics": metrics,
        "percentage_reductions_vs_multilingual_baselines": percentage_reductions,
        "bootstrap_confidence_intervals": bootstrap,
        "historical_8k_compatibility": eight_k_compatibility,
        "tradeoff_rows": tradeoff_rows,
        "operating_point": operating_point,
        "chart_contract": {
            "analytical_question": (
                "How does balanced mixed-BPE fertility change with vocabulary size "
                "across ASR and formal Twi?"
            ),
            "takeaway": (
                f"The fixed plateau rule selects {selected:,}; the chart shows both "
                "domain curves and exact fertility labels."
            ),
            "family": "ordered comparison",
            "variant": "two-series line with markers and direct labels",
            "data_sufficiency": (
                "Four pre-specified controlled operating points; an experimental curve, "
                "not a temporal trend."
            ),
            "palette": {
                "policy": "hard two-root cap",
                "asr": "#1f5a94",
                "formal": "#b27700",
                "non_color_distinction": "solid/filled versus dashed/open markers",
            },
            "surface": "standalone static SVG and PNG for paper and repository export",
        },
        "sources": {
            "hidden_dimensions": {
                "Qwen/Qwen3-0.6B-Base": (
                    "https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json"
                ),
                "Qwen/Qwen3-1.7B-Base": (
                    "https://huggingface.co/Qwen/Qwen3-1.7B-Base/blob/main/config.json"
                ),
                "google/gemma-3-1b-pt": (
                    "https://huggingface.co/google/gemma-3-1b-pt/blob/main/config.json"
                ),
                "meta-llama/Llama-3.2-1B": (
                    "https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json"
                ),
                "CohereLabs/tiny-aya-base": (
                    "https://huggingface.co/CohereLabs/tiny-aya-base/blob/main/config.json"
                ),
            }
        },
    }
    write_json(args.output_json, payload)
    ensure_parent_dir(args.output_table)
    args.output_table.write_text(render_tradeoff_markdown(payload), encoding="utf-8")
    save_fertility_figure(
        payload,
        svg_path=args.output_figure,
        png_path=args.output_figure_png,
    )
    print(f"Ablation JSON written to {args.output_json}")
    print(f"Trade-off table written to {args.output_table}")
    print(f"Fertility figure written to {args.output_figure}")
    print(f"Fertility PNG written to {args.output_figure_png}")
    print(f"Selected operating point: {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
