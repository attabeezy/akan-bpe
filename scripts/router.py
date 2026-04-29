#!/usr/bin/env python3
"""Router CLI for Akan-BPE tokenizer selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akan_bpe.router import AkanBPERouter, load_router_config, save_router_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Router CLI for Akan-BPE.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize router config")
    init_parser.add_argument("--asr-tokenizer", required=True)
    init_parser.add_argument("--tts-tokenizer", required=True)
    init_parser.add_argument("--mixed-tokenizer")
    init_parser.add_argument("--output", required=True)

    route_parser = subparsers.add_parser("route", help="Route a single text")
    route_parser.add_argument("--config", required=True)
    route_parser.add_argument("--text", required=True)
    route_parser.add_argument("--use-ml", action="store_true", help="Use ML classifier")

    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize text with router")
    tokenize_parser.add_argument("--config", required=True)
    tokenize_parser.add_argument("--text", required=True)
    tokenize_parser.add_argument("--show-decision", action="store_true")
    tokenize_parser.add_argument("--use-ml", action="store_true", help="Use ML classifier")

    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark router vs direct")
    benchmark_parser.add_argument("--config", required=True)
    benchmark_parser.add_argument("--test-file", required=True)
    benchmark_parser.add_argument("--output", required=True)
    benchmark_parser.add_argument("--max-samples", type=int, default=None)
    benchmark_parser.add_argument(
        "--use-ml", action="store_true", help="Use ML classifier instead of heuristic"
    )

    train_parser = subparsers.add_parser("train", help="Train ML classifier")
    train_parser.add_argument("--asr-train", required=True)
    train_parser.add_argument("--tts-train", required=True)
    train_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "init":
        config = {
            "asr_tokenizer_path": args.asr_tokenizer,
            "tts_tokenizer_path": args.tts_tokenizer,
            "mixed_tokenizer_path": args.mixed_tokenizer,
        }
        save_router_config(Path(args.output), config)
        print(f"Router config saved to {args.output}")

    elif args.command == "route":
        config = load_router_config(args.config)
        router = AkanBPERouter(
            asr_tokenizer_path=config["asr_tokenizer_path"],
            tts_tokenizer_path=config["tts_tokenizer_path"],
            mixed_tokenizer_path=config.get("mixed_tokenizer_path"),
            use_ml_classifier=args.use_ml if hasattr(args, "use_ml") else False,
            classifier_path=config.get("classifier_path"),
        )
        decision = router.route(args.text)
        print(f"Selected: {decision.selected_tokenizer}")
        print(f"Domain: {decision.domain}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Reason: {decision.reason}")
        router.close()

    elif args.command == "tokenize":
        config = load_router_config(args.config)
        router = AkanBPERouter(
            asr_tokenizer_path=config["asr_tokenizer_path"],
            tts_tokenizer_path=config["tts_tokenizer_path"],
            mixed_tokenizer_path=config.get("mixed_tokenizer_path"),
            use_ml_classifier=args.use_ml if hasattr(args, "use_ml") else False,
            classifier_path=config.get("classifier_path"),
        )
        tokens, decision = (
            router.route(args.text) if args.show_decision else (router.tokenize(args.text)[0], None)
        )
        if args.show_decision:
            print(f"Tokens: {tokens}")
            print(f"Selected: {decision.selected_tokenizer}")
            print(f"Confidence: {decision.confidence:.2f}")
        else:
            print(tokens)
        router.close()

    elif args.command == "benchmark":
        import json

        from akan_bpe.datasets import load_jsonl_samples
        from akan_bpe.classifier import MLClassifierRouter

        config = load_router_config(args.config)
        use_ml = getattr(args, "use_ml", False)
        classifier_path = config.get("classifier_path") if use_ml else None

        router = AkanBPERouter(
            asr_tokenizer_path=config["asr_tokenizer_path"],
            tts_tokenizer_path=config["tts_tokenizer_path"],
            mixed_tokenizer_path=config.get("mixed_tokenizer_path"),
            use_ml_classifier=use_ml,
            classifier_path=classifier_path,
        )

        samples = list(load_jsonl_samples(Path(args.test_file)))
        if args.max_samples:
            samples = samples[: args.max_samples]

        asr_count = 0
        tts_count = 0
        mixed_count = 0

        for sample in samples:
            text = sample.text
            decision = router.route(text)
            if decision.selected_tokenizer == "asr":
                asr_count += 1
            elif decision.selected_tokenizer == "tts":
                tts_count += 1
            else:
                mixed_count += 1

        result = {
            "config": args.config,
            "test_file": args.test_file,
            "total_samples": len(samples),
            "routing_decisions": {
                "asr": asr_count,
                "tts": tts_count,
                "mixed": mixed_count,
            },
            "percentages": {
                "asr": asr_count / len(samples) * 100 if samples else 0,
                "tts": tts_count / len(samples) * 100 if samples else 0,
                "mixed": mixed_count / len(samples) * 100 if samples else 0,
            },
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"Benchmark written to {args.output}")
        print(f"ASR: {asr_count} ({result['percentages']['asr']:.1f}%)")
        print(f"TTS: {tts_count} ({result['percentages']['tts']:.1f}%)")
        print(f"Mixed: {mixed_count} ({result['percentages']['mixed']:.1f}%)")
        router.close()

    elif args.command == "train":
        from akan_bpe.classifier import MLClassifierRouter

        classifier = MLClassifierRouter()
        result = classifier.train(
            asr_path=args.asr_train,
            tts_path=args.tts_train,
            output_path=args.output,
        )
        print(f"Classifier trained!")
        print(f"ASR samples: {result['asr_samples']}")
        print(f"TTS samples: {result['tts_samples']}")
        print(f"Total samples: {result['total_samples']}")
        print(f"Train accuracy: {result['train_accuracy']:.4f}")
        print(f"Saved to: {result['output_path']}")


if __name__ == "__main__":
    main()
