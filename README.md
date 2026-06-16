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

# 2. Train the balanced mixed Akan tokenizer
python scripts/train_bpe.py --inputs data/aka_asr_train.jsonl data/pristine_twi_train.jsonl \
    --output models/mixed_tokenizer.json --name mixed --balance

# 3. Fine-tune a base LLM with the mixed Akan tokenizer (QLoRA on a T4)
python scripts/model_integration.py --model-id Qwen/Qwen3-0.6B-Base
```

Step 3 derives sensible defaults (tokenizer, data, output paths, eval cap, LoRA config) from the
model id — a real run needs only `--model-id`. Model integration now defaults to the balanced
`models/mixed_tokenizer.json` tokenizer, replacing the earlier TTS-only integration tokenizer.
Add `--embedding-init-mode mean_subword` for the warm-start ablation, or `--device-mode smoke`
for a tiny CPU pipeline check.

`scripts/benchmark_fertility.py` compares the tokenizers against multilingual baselines and
`scripts/router.py` trains the domain router. Every script takes `--help` for its full options.

The ladder notebooks are split into `notebooks/run-full-light.ipynb` and
`notebooks/run-full-heavy.ipynb`. New mixed-tokenizer reruns write `-mixed` result IDs such as
`run-qwen-0.6b-mixed` and `run-qwen-0.6b-mixed-meansub` so previous TTS-tokenizer artifacts stay
available.

## Metric

- **Fertility** `F = total_tokens / total_words` — the intrinsic tokenizer metric (lower is better).
- **Bits-per-byte (BPB)** — the cross-tokenizer modeling metric. Perplexity is not comparable
  across vocabularies, so BPB normalizes each model's total NLL by the fixed UTF-8 byte count of
  the eval text. Scoring uses **full byte coverage** (every byte of every text), so high-fertility
  base tokenizers are compared honestly against the Akan tokenizer.
- **chrF / chrF++** — held-out Twi continuation quality for M4 generation evaluation. The
  executed notebooks score 512 examples per arm using 48 prompt words, 64 reference words, and
  64 generated tokens.

## Model Ladder Results

The executed split notebooks are the source of truth for the current 5-model QLoRA ladder:
`notebooks/run-full-light.ipynb` and `notebooks/run-full-heavy.ipynb`. The derived combined
artifact is `results/notebook-ladder-results.json`, generated with:

```bash
python scripts/extract_notebook_results.py
```

The artifact preserves the full notebook payloads plus a flattened summary. In the notebook-derived
results, `mean_subword` improves chrF and chrF++ over `random` in all five model runs, beats
`random` on BPB in all five runs, and beats each base model on BPB in all five mean-subword arms.
The older `results/model-ladder-results.json` remains as a previous baseline artifact, not the
current source of truth.

Note: the checked-in ladder result artifacts were generated before the integration default moved
from the TTS tokenizer to the balanced mixed tokenizer. They remain available for comparison and
will be replaced by mixed-tokenizer result artifacts once the reruns complete.

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
- [x] Generation quality (chrF on held-out Twi)
- [ ] IEEE Ghana ICAST 2026 write-up

## License

[MIT](LICENSE).
