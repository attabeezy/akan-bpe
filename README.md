# somax

**Tokenizer-only Twi experiments for studying the Tokenization Tax**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Modern LLM tokenizers are optimized for English, resulting in a **Tokenization Tax**
for languages like Twi. SOMAX keeps the first phase deliberately small:

- normalize Twi ASR and formal-text datasets
- train tokenizer variants for `asr`, `tts`, and `mixed`
- compare them against a baseline tokenizer in one unified fertility experiment JSON

## Data Sources

| Stream | Source | Notes |
|--------|--------|-------|
| **ASR** | `google/WaxalNLP` - `aka_asr` | Noisy Twi/Akan speech transcriptions |
| **Formal** | `ghananlpcommunity/pristine-twi-english` | Clean, structured Twi text |

## Active Components

- `scripts/download.py` - download and normalize Twi datasets into `data/`
- `scripts/train_bpe.py` - train one tokenizer variant per run
- `scripts/benchmark_fertility.py` - compare tokenizers in one unified experiment
- `somax/` - thin helpers for JSONL loading, tokenizer training, fertility metrics, and experiment assembly

## Quick Start

### Prerequisites

```bash
pip install -e ".[dev]"
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
    --name mixed

# 3. Run one unified fertility experiment
python scripts/benchmark_fertility.py \
    --experiment-id tokenizer_fertility_experiment_001 \
    --control-tokenizer gpt2 \
    --asr-tokenizer models/asr_tokenizer.json \
    --tts-tokenizer models/tts_tokenizer.json \
    --mixed-tokenizer models/mixed_tokenizer.json \
    --asr-test-file data/aka_asr_test.jsonl \
    --tts-test-file data/pristine_twi_test.jsonl \
    --output results/tokenizer_fertility_experiment_001.json
```

## Tokenizer Variants

| Variant | Training Corpus | Purpose |
|--------|------------------|---------|
| `control` | Existing pretrained tokenizer | Baseline reference |
| `asr` | `aka_asr_train.jsonl` | Specialized conversational tokenizer |
| `tts` | `pristine_twi_train.jsonl` | Specialized formal tokenizer |
| `mixed` | both training files | Single-tokenizer compromise |

## Metric

Primary metric:

```text
F = total_tokens / total_words
```

Lower fertility means the tokenizer needs fewer tokens per word on the same text.

## Output Contract

SOMAX keeps experiment output simple:

- one tokenizer training run writes one tokenizer artifact and one optional stats JSON
- one benchmark experiment writes one unified JSON

Example outputs:

- `models/asr_tokenizer.json`
- `models/asr_tokenizer_stats.json`
- `results/tokenizer_fertility_experiment_001.json`

The unified experiment JSON contains:

- experiment metadata
- tokenizer references
- ASR and TTS test-set paths
- fertility results for every tokenizer on every test set
- a small summary of which tokenizer wins where

## Project Structure

```text
SOMAX/
├── data/                  # Twi datasets (gitignored)
├── models/                # Tokenizer artifacts (gitignored)
├── results/               # Unified experiment outputs
├── scripts/
│   ├── download.py
│   ├── train_bpe.py
│   └── benchmark_fertility.py
├── somax/
├── tests/
├── pyproject.toml
├── Makefile
└── README.md
```

## Roadmap

- [x] Dataset download and normalization
- [x] Tokenizer-only project reference
- [x] Unified one-experiment-one-JSON benchmark flow
- [ ] Compare `asr`, `tts`, and `mixed` against a baseline tokenizer
- [ ] Decide whether specialization is strong enough to justify router or mux work later

## License

This project is licensed under the MIT License.
