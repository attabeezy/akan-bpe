# akan-bpe

**Akan tokenizer experiments with Phase 2A model-integration scaffolding**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Modern LLM tokenizers are optimized for English, resulting in a **Tokenization Tax**
for languages like Akan. Akan-BPE completed the tokenizer-only phase and has now
landed its first Phase 2A model-integration run:

- normalize Akan ASR and formal-text datasets
- train tokenizer variants for `asr`, `tts`, and `mixed`
- compare them against multilingual baselines (XLM-R, mBERT, mT5) in one unified fertility experiment JSON
- fine-tune `Qwen/Qwen3-0.6B` with the Akan TTS tokenizer via QLoRA on Colab/T4
  (Phase 2A1, completed): **49.5% fewer tokens/word** than the base tokenizer on the
  eval set, coherent Twi generation (cross-tokenizer modeling claim moves to bits-per-byte
  from Phase 2A2 — see Roadmap)

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

`smoke` mode is a tiny-model pipeline validation run. It checks tokenizer loading,
dataset preparation, embedding resize, a forward/eval pass, and generation, then
writes the result JSON without performing full fine-tuning or saving adapters.

`colab-qlora` is currently an explicit 2A1 path for `Qwen/Qwen3-0.6B` on Colab/T4.
It performs adapter training, eval, generation, saves adapters plus tokenizer, and
verifies the saved output can be reloaded for inference.

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
│   ├── experiment.py
│   ├── datasets.py
│   ├── io.py
│   └── model_integration.py
├── tests/
├── notebooks/
│   ├── train_eval.ipynb              # End-to-end walkthrough
│   └── phase2a_qwen3_tts_colab.ipynb
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
- [x] Phase 2A1: Qwen3-0.6B QLoRA fine-tune with Akan TTS tokenizer on Colab/T4 (49.5% fertility reduction)
- [ ] **M2** methodology hardening: bits-per-byte (BPB) metric, embedding-init ablation, regenerated ASR test split
- [ ] **M3** model evidence — 5 runs reported in BPB (span scale + family + base-vocab + multilinguality):
  - [x] 2A1 `Qwen/Qwen3-0.6B` (scale anchor, low)
  - [ ] 2A2 `Qwen/Qwen3-1.7B` (scale anchor, high)
  - [ ] 2A3 `google/gemma-3-1b-pt` (multilingual, 256k vocab)
  - [ ] 2A4 `meta-llama/Llama-3.2-1B` (English-centric, deployment-standard)
  - [ ] 2A5 `CohereLabs/tiny-aya-base` (Africa-aware, 3.35B — run last; custom arch)
- [ ] **M4** generation quality: chrF on held-out Twi
- [ ] **M5** write & submit (AfricaNLP / WiNLP workshop)
- [ ] _(future work)_ 2A6 stretch tier (Phi-4-mini, aya-expanse-8b), edge deployment benchmark on hardware

This roadmap is driven by an **AfricaNLP / WiNLP workshop** submission. See `project.md` §0
(Research Design & Road to Paper) for the locked decisions, milestones, and critical path.

### Next up: M2 — methodology hardening (before more model runs)

The single highest-leverage item is **bits-per-byte (BPB)**: raw perplexity is not comparable
across tokenizers with different vocabularies, so the cross-tokenizer claim needs a
tokenizer-agnostic metric in `akan_bpe/model_integration.py`. Land BPB, the embedding-init
ablation, and the regenerated ASR test split *before* running 2A2 — otherwise the model runs
must be redone. Then proceed to Phase 2A2:

1. Add `"Qwen/Qwen3-1.7B"` to `SUPPORTED_COLAB_QLORA_MODEL_IDS` in
   `akan_bpe/model_integration.py` (the `colab-qlora` allowlist is currently pinned to
   Qwen3-0.6B) and extend the matching test in `tests/test_model_integration.py`.
2. Clone `notebooks/phase2a_qwen3_tts_colab.ipynb`, point `--model-id` at `Qwen/Qwen3-1.7B`, and
   keep the TTS tokenizer. 1.7B in 4-bit fits a free T4 but may need a smaller
   `--batch-size` / larger `--grad-accum` than 2A1.
3. Record the same metrics (now including BPB) so 2A1 vs 2A2 is an apples-to-apples comparison.

See `project.md` §0.3 and §16.1.1 for the full checklist.

## Notes & Limitations

- **Scripts require the dev install.** `scripts/*.py` import `python-dotenv` and the
  `akan_bpe` package, so run `pip install -e ".[dev]"` (add `,train` for model
  integration) before invoking them — a bare interpreter will fail on import.
- **The ASR test split is a single sample.** It is a stale-download artifact, not a code
  bug; ASR-test fertility numbers are therefore anecdotal. This is an **in-scope fix (M2)** for
  the workshop paper, which keeps the dual-regime story: regenerate a proper ASR test split with
  `scripts/download.py` and re-run the fertility benchmark before the model experiments. Until
  then, the formal (TTS) test set (2,500 samples) carries the statistical weight.
- **Cross-tokenizer perplexity is not comparable.** From Phase 2A2 onward the modeling claim
  uses **bits-per-byte (BPB)** plus **chrF**, not raw perplexity across different tokenizers.

## License

This project is licensed under the MIT License.
