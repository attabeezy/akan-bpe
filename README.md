# akan-bpe

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Specialized BPE tokenizers for **Akan (Twi)** that cut the "tokenization tax" English-centric
LLM tokenizers impose on low-resource languages. The repo trains Akan tokenizers, benchmarks
their fertility against multilingual baselines, and fine-tunes base LLMs with the Akan
tokenizer via QLoRA — measuring the gain in **bits-per-byte**, a tokenizer-agnostic metric.

## Data Sources

| Stream | Source | Notes |
|--------|--------|-------|
| **ASR** | `google/WaxalNLP` — `aka_asr` | Noisy Akan speech transcriptions |
| **Formal** | `ghananlpcommunity/pristine-twi-english` | Clean, structured Akan text |

## Install

```bash
uv pip install -e ".[dev]"            # library + tooling
uv pip install -e ".[dev,train]"      # add for model integration (QLoRA)
uv pip install bitsandbytes           # add for Colab/Kaggle QLoRA runs
```

Scripts import the `akan_bpe` package, so install it before running them.

## Quick Start

```bash
# 1. Download and normalize datasets into data/
python scripts/download.py --output-dir data

# 2. Train the Akan TTS tokenizer
python scripts/train_bpe.py --inputs data/pristine_twi_train.jsonl \
    --output models/tts_tokenizer.json --name tts

# 3. Fine-tune a base LLM with the Akan tokenizer (QLoRA on a T4)
python scripts/model_integration.py --model-id Qwen/Qwen3-0.6B-Base
```

Step 3 derives sensible defaults (tokenizer, data, output paths, eval cap, LoRA config) from the
model id — a real run needs only `--model-id`. Add `--embedding-init-mode mean_subword` for the
warm-start ablation, or `--device-mode smoke` for a tiny CPU pipeline check.

`scripts/benchmark_fertility.py` compares the tokenizers against multilingual baselines and
`scripts/router.py` trains the domain router. Every script takes `--help` for its full options.

The five ladder runs each have a ready-to-run notebook: `notebooks/run-<model>.ipynb`
(`run-qwen-0.6b`, `run-qwen-1.7b`, `run-gemma-1b`, `run-llama-1b`, `run-aya-base`).

## Metric

- **Fertility** `F = total_tokens / total_words` — the intrinsic tokenizer metric (lower is better).
- **Bits-per-byte (BPB)** — the cross-tokenizer modeling metric. Perplexity is not comparable
  across vocabularies, so BPB normalizes each model's total NLL by the fixed UTF-8 byte count of
  the eval text. Scoring uses **full byte coverage** (every byte of every text), so high-fertility
  base tokenizers are compared honestly against the Akan tokenizer.

## Model Ladder Results

The 5-model QLoRA ladder has been re-scored with the corrected full-coverage BPB metric. Full
run payloads and a combined machine-readable artifact are tracked in
`results/model-ladder-results.json`; the source notebooks preserve the original printed JSON
blocks used for recovery.

| Model | Arm | Base fert. | Akan fert. | Token reduction | Base BPB | Akan BPB | BPB improvement | Perplexity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen-0.6b` | `random` | 2.530 | 1.259 | 50.3% | 2.9523 | 1.4805 | +1.4718 | 82.68 |
| `qwen-0.6b` | `mean_subword` | 2.530 | 1.259 | 50.3% | 2.9523 | 1.2832 | +1.6691 | 45.28 |
| `qwen-1.7b` | `random` | 2.530 | 1.259 | 50.3% | 2.7556 | 1.4741 | +1.2815 | 82.36 |
| `qwen-1.7b` | `mean_subword` | 2.530 | 1.259 | 50.3% | 2.7556 | 1.2478 | +1.5078 | 40.93 |
| `gemma-1b` | `random` | 2.284 | 1.259 | 44.9% | 3.3907 | 1.4841 | +1.9067 | 87.14 |
| `gemma-1b` | `mean_subword` | 2.284 | 1.259 | 44.9% | 3.3907 | 1.2355 | +2.1553 | 39.27 |
| `llama-1b` | `random` | 3.073 | 1.259 | 59.0% | 2.4480 | 1.4150 | +1.0330 | 69.09 |
| `llama-1b` | `mean_subword` | 3.073 | 1.259 | 59.0% | 2.4480 | 1.2337 | +1.2143 | 39.24 |
| `aya-base` | `random` | 2.975 | 1.259 | 57.7% | 2.7129 | 1.4755 | +1.2374 | 82.49 |
| `aya-base` | `mean_subword` | 2.975 | 1.259 | 57.7% | 2.7129 | 1.2323 | +1.4806 | 38.84 |

## Project Structure

```text
akan-bpe/
├── akan_bpe/        # core library (tokenizers, metrics, router, model integration)
├── scripts/         # download, train_bpe, benchmark_fertility, router, model_integration
├── notebooks/       # run-<model>.ipynb ladder runs + train_eval walkthrough
├── results/         # tracked JSON experiment outputs and combined ladder artifact
├── tests/
└── pyproject.toml
```

Datasets and large model artifacts are generated locally and gitignored.

## Roadmap

- [x] Akan ASR / TTS / mixed tokenizers + fertility benchmark vs XLM-R, mBERT, mT5
- [x] Heuristic + ML-classifier domain router (held-out eval)
- [x] Model-integration ladder — 5 QLoRA runs across 4 families on Kaggle/T4
- [x] Bits-per-byte metric with full byte coverage + mean-of-subword embedding-init ablation
- [x] Re-score the ladder under the corrected BPB metric
- [ ] Generation quality (chrF on held-out Twi)
- [ ] Workshop write-up (AfricaNLP / WiNLP)

## License

[MIT](LICENSE).
