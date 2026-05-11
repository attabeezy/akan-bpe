# akan-bpe

**Akan tokenizer experiments with Phase 2A model-integration scaffolding**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Modern LLM tokenizers are optimized for English, resulting in a **Tokenization Tax**
for languages like Akan. Akan-BPE completed the tokenizer-only phase and now has
the first Phase 2A scaffold in place:

- normalize Akan ASR and formal-text datasets
- train tokenizer variants for `asr`, `tts`, and `mixed`
- compare them against multilingual baselines (XLM-R, mBERT, mT5) in one unified fertility experiment JSON
- run the first model-integration path for `Qwen/Qwen3-0.6B` with the Akan TTS tokenizer

## Data Sources

| Stream | Source | Notes |
|--------|--------|-------|
| **ASR** | `google/WaxalNLP` - `aka_asr` | Noisy Akan speech transcriptions |
| **Formal** | `ghananlpcommunity/pristine-twi-english` | Clean, structured Akan text |

## Active Components

- `scripts/download.py` - download and normalize Akan datasets into `data/`
- `scripts/train_bpe.py` - train one tokenizer variant per run
- `scripts/benchmark_fertility.py` - compare tokenizers in one unified experiment
- `scripts/router.py` - train ML classifier and benchmark routing strategies
- `scripts/model_integration.py` - run one model-integration experiment and write one result JSON
- `akan_bpe/` - thin helpers for JSONL loading, tokenizer training, fertility metrics, router, classifier, and model integration

## Quick Start

### Prerequisites

```bash
pip install -e ".[dev]"
pip install sentencepiece   # required for mT5 tokenizer
```

For Phase 2A model integration:

```bash
pip install -e ".[dev,train]"
pip install bitsandbytes    # required for Colab QLoRA runs
```

### Run Locally

```bash
# 1. Download datasets
python scripts/download.py --output-dir data

# 2. Train tokenizer variants
python scripts/train_bpe.py \
    --inputs data/aka_asr_train.jsonl \
    --output models/asr_tokenizer.json \
    --name asr

python scripts/train_bpe.py \
    --inputs data/pristine_twi_train.jsonl \
    --output models/tts_tokenizer.json \
    --name tts

python scripts/train_bpe.py \
    --inputs data/aka_asr_train.jsonl data/pristine_twi_train.jsonl \
    --output models/mixed_tokenizer.json \
    --name mixed \
    --balance

# 3. Run one unified fertility experiment
python scripts/benchmark_fertility.py \
    --experiment-id tokenizer_fertility_experiment_001 \
    --baselines xlm-roberta-base bert-base-multilingual-cased google/mt5-base \
    --asr-tokenizer models/asr_tokenizer.json \
    --tts-tokenizer models/tts_tokenizer.json \
    --mixed-tokenizer models/mixed_tokenizer.json \
    --asr-test-file data/aka_asr_test.jsonl \
    --tts-test-file data/pristine_twi_test.jsonl \
    --output results/tokenizer_fertility_experiment_001.json

# 4. (Optional) Train ML router classifier
python scripts/router.py train \
    --asr-train data/aka_asr_train.jsonl \
    --tts-train data/pristine_twi_train.jsonl \
    --output models/router_classifier.pkl

# 5. Phase 2A1 model-integration scaffold
python scripts/model_integration.py \
    --experiment-id phase2a1_qwen3_0_6b_tts \
    --model-id Qwen/Qwen3-0.6B \
    --tokenizer-path models/tts_tokenizer.json \
    --train-file data/pristine_twi_train.jsonl \
    --eval-file data/pristine_twi_test.jsonl \
    --output-dir models/phase2a1_qwen3_0_6b_tts \
    --results-output results/phase2a1_qwen3_0_6b_tts.json \
    --device-mode smoke \
    --max-train-samples 64 \
    --max-eval-samples 32
```

## Tokenizer Variants

| Variant | Training Corpus | Purpose |
|--------|------------------|---------|
| `baselines` | XLM-R, mBERT, mT5 (pretrained) | Multilingual reference baselines |
| `asr` | `aka_asr_train.jsonl` | Specialized conversational tokenizer |
| `tts` | `pristine_twi_train.jsonl` | Specialized formal tokenizer |
| `mixed` | both files, corpus-balanced | Single-tokenizer compromise |

## Metric

Primary metric:

```text
F = total_tokens / total_words
```

Lower fertility means the tokenizer needs fewer tokens per word on the same text.

## Output Contract

Akan-BPE keeps experiment output simple:

- one tokenizer training run writes one tokenizer artifact and one optional stats JSON
- one benchmark experiment writes one unified JSON
- one model-integration run writes one model/adapters output directory and one unified JSON

Example outputs:

- `models/asr_tokenizer.json`
- `models/asr_tokenizer_stats.json`
- `results/tokenizer_fertility_experiment_001.json`
- `models/phase2a1_qwen3_0_6b_tts/`
- `results/phase2a1_qwen3_0_6b_tts.json`

The unified experiment JSON contains:

- experiment metadata
- tokenizer references
- ASR and TTS test-set paths
- fertility results for every tokenizer on every test set
- a small summary of which tokenizer wins where

The model-integration JSON contains:

- experiment metadata
- base model identifier and tokenizer path
- train/eval dataset paths and sample counts
- token-count comparison against the base model tokenizer
- eval loss and perplexity
- qualitative generation samples
- output model directory reference

## Project Structure

```text
Akan-BPE/
├── data/                  # Akan datasets (gitignored)
│   └── akan/              # Raw ASR downloads
├── models/                # Tokenizer + classifier artifacts (gitignored)
├── results/               # Experiment outputs (gitignored)
├── config/                # Router configuration
├── scripts/
│   ├── download.py
│   ├── train_bpe.py
│   ├── benchmark_fertility.py
│   ├── model_integration.py
│   └── router.py
├── akan_bpe/              # Core library
│   ├── tokenizers.py
│   ├── router.py
│   ├── classifier.py
│   ├── metrics.py
│   ├── datasets.py
│   └── model_integration.py
├── tests/
├── train_eval.ipynb       # End-to-end walkthrough
├── phase2a_qwen3_tts_colab.ipynb
├── report.md              # Technical report
├── pyproject.toml
├── Makefile
└── README.md
```

## Roadmap

- [x] Dataset download and normalization
- [x] Train ASR, TTS, and Mixed tokenizers
- [x] Fertility benchmark comparing all tokenizers vs baseline
- [x] Implement and benchmark heuristic router
- [x] Train ML classifier router (99.99% train/test accuracy)
- [x] Generate technical report (report.md)
- [x] Replace GPT-2 with multilingual baselines (XLM-R, mBERT, mT5)
- [x] Fix mixed tokenizer corpus imbalance (balanced upsampling)
- [x] Add held-out test evaluation to router classifier
- [x] Add 2A1 model-integration scaffold and Colab notebook
- [ ] Model integration (resize vocab, test generation)
- [ ] Edge deployment (benchmark on hardware)

## License

This project is licensed under the MIT License.
