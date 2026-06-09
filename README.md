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
- fine-tune base LLMs with the Akan TTS tokenizer via QLoRA on Kaggle/T4 across a **complete
  5-run model ladder** — `Qwen/Qwen3-0.6B` (2A1), `Qwen/Qwen3-1.7B` (2A2, scale),
  `google/gemma-3-1b-pt` (2A3, family + 256k vocab), `meta-llama/Llama-3.2-1B` (2A4,
  English-centric), `CohereLabs/tiny-aya-base` (2A5, Africa-aware): **~45–59% fewer tokens/word**
  than the base tokenizer, with coherent Twi generation. With mean-of-subword embedding init the
  fine-tuned model converges to **~0.90 bits-per-byte on all five rungs** (0.932 / 0.907 / 0.896 /
  0.897 / 0.895) while the *bases* range from 0.769 to 1.389. Key findings: at 1.7B the stronger
  base means *random* init no longer wins (only mean-of-subword does); the tokenization tax
  survives even Gemma's 256k multilingual vocab *and* an Africa-built base (not a small-vocab or
  single-family artifact). The two rungs where the fine-tune appears to "lose" on BPB (Llama,
  tiny-aya) are exactly the two highest-base-fertility runs — the fingerprint of a **truncation
  bias** in the metric (truncates at 256 tokens, divides by full bytes), under a corrected
  re-evaluation (`notebooks/2a4_llama_eval_helper.ipynb`)

> **Looking for the full plan?** [`project.md`](project.md) is the authoritative
> project reference — research design, milestones, model ladder, file contracts, and
> limitations. [`report.md`](report.md) is the technical report.

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

> **Scripts require the dev install.** `scripts/*.py` import `python-dotenv` and the
> `akan_bpe` package, so install the package (add `,train` for model integration)
> before invoking them — a bare interpreter will fail on import.

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

`smoke` mode is a tiny-model pipeline validation run (tokenizer loading, dataset prep,
embedding resize, a forward/eval pass, and generation) that writes the result JSON
without full fine-tuning. `colab-qlora` is the explicit 2A1 path for `Qwen/Qwen3-0.6B`
on Colab/T4: adapter training, eval, generation, save, and reload-for-inference verification.

Both modes report **bits-per-byte (BPB)** for the base model and the fine-tuned model on
the same eval bytes (`eval.bpb` in the result JSON) — a tokenizer-agnostic, honest
cross-tokenizer comparison. Two extra flags control the M2 hardening:

- `--embedding-init-mode {random,mean_subword}` — `mean_subword` initializes each Akan-vocab
  embedding from the mean of the base model's subword embeddings for that token's surface
  string (the modeling contribution; run `random` vs `mean_subword` as a clean A/B).
- `--skip-base-bpb` — skip the extra base-model BPB pass when GPU memory is tight.

## Tokenizer Variants

| Variant | Training Corpus | Purpose |
|--------|------------------|---------|
| `baselines` | XLM-R, mBERT, mT5 (pretrained) | Multilingual reference baselines |
| `asr` | `aka_asr_train.jsonl` | Specialized conversational tokenizer |
| `tts` | `pristine_twi_train.jsonl` | Specialized formal tokenizer |
| `mixed` | both files, corpus-balanced | Single-tokenizer compromise |

## Metric

Primary metric — **token fertility**:

```text
F = total_tokens / total_words
```

Lower fertility means the tokenizer needs fewer tokens per word on the same text.
From Phase 2A2 onward the cross-tokenizer modeling claim uses **bits-per-byte (BPB)**
plus **chrF**, not raw perplexity (see [`project.md`](project.md) §0).

## Output Contract

Akan-BPE keeps experiment output simple — one run, one set of artifacts:

- one tokenizer training run → one tokenizer artifact (+ optional stats JSON)
- one benchmark experiment → one unified JSON
- one model-integration run → one model/adapters directory + one unified JSON

See [`project.md`](project.md) §11 for the full file contracts and JSON field lists.

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
│   └── 2a1_qwen3-0.6b_tts.ipynb
├── report.md              # Technical report
├── project.md             # Full project reference & paper plan
├── pyproject.toml
├── Makefile
└── README.md
```

## Roadmap

- [x] Dataset download and normalization
- [x] Train ASR, TTS, and Mixed tokenizers
- [x] Fertility benchmark vs multilingual baselines (XLM-R, mBERT, mT5)
- [x] Heuristic router + ML classifier router (held-out eval)
- [x] Technical report (`report.md`)
- [x] Phase 2A1: Qwen3-0.6B QLoRA fine-tune with Akan TTS tokenizer on Kaggle/T4 (50.3% fertility reduction)
- [x] Phase 2A2: Qwen3-1.7B QLoRA scale step on Kaggle/T4 (mean-subword BPB 0.907 vs base 1.031; random-init loses to base at this scale)
- [x] Phase 2A3: Gemma-3-1B QLoRA family + 256k-vocab step on Kaggle/T4 (44.9% fertility reduction; mean-subword BPB 0.896 vs weak base 1.389 — largest per-byte win; not a Qwen quirk)
- [x] Phase 2A4: Llama-3.2-1B QLoRA English-centric step on Kaggle/T4 (59.0% fertility reduction — biggest tax; first negative BPB, 0.897 vs unusually low base 0.769 — very likely a truncation artifact, see report §11)
- [x] Phase 2A5: tiny-aya-base QLoRA Africa-aware step on Kaggle/T4 — **final rung, M3 complete** (57.7% fertility reduction; mean-subword 0.895 vs base 0.878, −0.017 break-even; one of the two highest-base-fertility rungs that fingerprint the truncation bias)
- [ ] Truncation-corrected BPB re-score (`notebooks/2a4_llama_eval_helper.ipynb`) — open item before per-byte numbers are final; fertility + ablation results are unaffected
- [x] **M2** methodology hardening:
  - [x] bits-per-byte (BPB) metric — base vs experiment model on the same eval bytes (`akan_bpe/model_integration.py`)
  - [x] embedding-init ablation — `--embedding-init-mode {random,mean_subword}`; on 2A1, mean-of-subword init wins (BPB 0.932 vs random 1.079, perplexity 45.4 vs 82.7) and is now the ladder default
  - [x] regenerated ASR test split — full WaxalNLP stream re-split to 8,085/1,011/1,011 (was a 1-sample test); ASR + mixed tokenizers and router retrained, benchmark re-run; `scripts/download.py` now fails loudly on a truncated split
- [x] **M3** model evidence: 5 runs across families/scales (Qwen3-0.6B ✅, Qwen3-1.7B ✅, Gemma-3-1B ✅, Llama-3.2-1B ✅, tiny-aya-base ✅) reported in BPB — **complete**
- [ ] **M4** generation quality: chrF on held-out Twi
- [ ] **M5** write & submit (AfricaNLP / WiNLP workshop)

This roadmap is driven by an **AfricaNLP / WiNLP workshop** submission. The full plan —
locked decisions, the five-model ladder, the critical path, and the next concrete actions
for M2/2A2 — lives in [`project.md`](project.md) §0 and §16.

## License

This project is licensed under the MIT License.
