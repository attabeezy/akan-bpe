#!/usr/bin/env python3
"""Benchmark script for edge device latency and token fertility.

Measures performance metrics for WAXAL-Dual-Core on Dell Latitude 7400 (8GB RAM):
- Token Fertility (F = Tokens / Words) - primary metric
- Tokens per second (TPS)
- Memory usage
- Inference latency

Usage:
    # With GGUF model
    python benchmark_edge.py --model models/gguf/model-Q4_K_M.gguf --test-file data/akan/test.jsonl

    # With HuggingFace model
    python benchmark_edge.py --model meta-llama/Llama-3.2-1B --test-file data/akan/test.jsonl --huggingface

    # Baseline (no model, just tokenizer)
    python benchmark_edge.py --tokenizer meta-llama/Llama-3.2-1B --test-file data/akan/test.jsonl --baseline
"""

import argparse
import json
import time
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    fertility: float
    tokens_per_second: float
    latency_seconds: float
    total_tokens: int
    total_words: int
    num_samples: int
    memory_mb: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "fertility": self.fertility,
            "tokens_per_second": self.tokens_per_second,
            "latency_seconds": self.latency_seconds,
            "total_tokens": self.total_tokens,
            "total_words": self.total_words,
            "num_samples": self.num_samples,
            "memory_mb": self.memory_mb,
        }


def calculate_fertility(text: str, tokens: list[int]) -> tuple[int, int, float]:
    """Calculate Token Fertility metric.

    F = Total Tokens / Total Words

    Target: Reduce F by >= 30% compared to baseline.

    Args:
        text: Original input text.
        tokens: Tokenized output.

    Returns:
        Tuple of (word_count, token_count, fertility_ratio).
    """
    words = len(text.split())
    if words == 0:
        return 0, 0, 0.0
    token_count = len(tokens)
    fertility = token_count / words
    return words, token_count, fertility


def load_test_texts(test_file: Path, max_samples: int = 100) -> list[str]:
    """Load test texts from JSONL file.

    Args:
        test_file: Path to JSONL test file.
        max_samples: Maximum number of samples to load.

    Returns:
        List of text strings.
    """
    texts = []
    with open(test_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                item = json.loads(line.strip())
                text = item.get("transcription") or item.get("text", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def benchmark_huggingface(model_id: str, texts: list[str]) -> BenchmarkResult:
    """Benchmark with HuggingFace model.

    Args:
        model_id: HuggingFace model ID.
        texts: List of text samples.

    Returns:
        Benchmark results.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers and torch required. Install with: pip install transformers torch"
        )

    print(f"Loading HuggingFace model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    total_tokens = 0
    total_words = 0
    latencies = []

    print(f"Running benchmark on {len(texts)} samples...")

    for text in texts:
        words, tokens, _ = calculate_fertility(text, [])
        encodings = tokenizer(text, return_tensors="pt")
        token_count = encodings["input_ids"].shape[1]

        total_tokens += token_count
        total_words += words

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(encodings["input_ids"], max_new_tokens=10, do_sample=False)
        latency = time.time() - start
        latencies.append(latency)

    avg_fertility = total_tokens / total_words if total_words > 0 else 0
    avg_latency = statistics.mean(latencies)
    tps = total_tokens / sum(latencies) if sum(latencies) > 0 else 0

    memory_mb = None
    try:
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    return BenchmarkResult(
        fertility=avg_fertility,
        tokens_per_second=tps,
        latency_seconds=avg_latency,
        total_tokens=total_tokens,
        total_words=total_words,
        num_samples=len(texts),
        memory_mb=memory_mb,
    )


def benchmark_gguf(model_path: Path, texts: list[str]) -> BenchmarkResult:
    """Benchmark with GGUF model (llama-cpp-python).

    Args:
        model_path: Path to GGUF model file.
        texts: List of text samples.

    Returns:
        Benchmark results.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("llama-cpp-python required. Install with: pip install llama-cpp-python")

    print(f"Loading GGUF model: {model_path}")

    llm = Llama(model_path=str(model_path), n_ctx=2048, n_threads=4, verbose=False)

    total_tokens = 0
    total_words = 0
    latencies = []

    print(f"Running benchmark on {len(texts)} samples...")

    for text in texts:
        words = len(text.split())
        tokens = llm.tokenize(text.encode("utf-8"))
        token_count = len(tokens)

        total_tokens += token_count
        total_words += words

        start = time.time()
        output = llm(text, max_tokens=10, temperature=0.0)
        latency = time.time() - start
        latencies.append(latency)

    avg_fertility = total_tokens / total_words if total_words > 0 else 0
    avg_latency = statistics.mean(latencies)
    tps = total_tokens / sum(latencies) if sum(latencies) > 0 else 0

    memory_mb = None
    try:
        import psutil

        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    return BenchmarkResult(
        fertility=avg_fertility,
        tokens_per_second=tps,
        latency_seconds=avg_latency,
        total_tokens=total_tokens,
        total_words=total_words,
        num_samples=len(texts),
        memory_mb=memory_mb,
    )


def benchmark_tokenizer(model_id: str, texts: list[str]) -> BenchmarkResult:
    """Benchmark with just the tokenizer (baseline).

    Args:
        model_id: HuggingFace model/tokenizer ID.
        texts: List of text samples.

    Returns:
        Benchmark results.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("transformers required. Install with: pip install transformers")

    print(f"Loading tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    total_tokens = 0
    total_words = 0
    latencies = []

    print(f"Running baseline benchmark on {len(texts)} samples...")

    for text in texts:
        words = len(text.split())
        start = time.time()
        tokens = tokenizer.encode(text)
        latency = time.time() - start

        token_count = len(tokens)
        total_tokens += token_count
        total_words += words
        latencies.append(latency)

    avg_fertility = total_tokens / total_words if total_words > 0 else 0
    avg_latency = statistics.mean(latencies)
    tps = total_tokens / sum(latencies) if sum(latencies) > 0 else 0

    return BenchmarkResult(
        fertility=avg_fertility,
        tokens_per_second=tps,
        latency_seconds=avg_latency,
        total_tokens=total_tokens,
        total_words=total_words,
        num_samples=len(texts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge device benchmark")
    parser.add_argument("--model", type=str, help="Model path or HuggingFace ID")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer ID (for baseline)")
    parser.add_argument("--test-file", type=str, required=True, help="JSONL test file")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to test")
    parser.add_argument("--huggingface", action="store_true", help="Use HuggingFace model")
    parser.add_argument("--baseline", action="store_true", help="Baseline tokenizer only")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not args.model and not args.tokenizer:
        parser.error("Either --model or --tokenizer is required")

    test_file = Path(args.test_file)
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    texts = load_test_texts(test_file, args.max_samples)
    if not texts:
        raise ValueError(f"No texts loaded from {test_file}")

    print(f"Loaded {len(texts)} test samples")
    print()

    if args.baseline and args.tokenizer:
        result = benchmark_tokenizer(args.tokenizer, texts)
    elif args.huggingface and args.model:
        result = benchmark_huggingface(args.model, texts)
    elif args.model:
        model_path = Path(args.model)
        if model_path.exists():
            result = benchmark_gguf(model_path, texts)
        else:
            result = benchmark_huggingface(args.model, texts)
    else:
        parser.error("Invalid combination of arguments")

    print("\n=== Benchmark Results ===")
    print(f"Token Fertility (F): {result.fertility:.2f} tokens/word")
    print(f"Tokens/second: {result.tokens_per_second:.2f}")
    print(f"Avg Latency: {result.latency_seconds:.4f}s")
    print(f"Total tokens: {result.total_tokens}")
    print(f"Total words: {result.total_words}")
    print(f"Samples: {result.num_samples}")
    if result.memory_mb:
        print(f"Memory: {result.memory_mb:.0f} MB")

    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
