# akan-bpe

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Specialized BPE tokenizers for **Akan (Twi)** that reduce the tokenization tax English-centric LLMs impose on low-resource languages, benchmarked via fertility and bits-per-byte across a 5-model QLoRA ladder.

## Install

```bash
uv pip install -e ".[dev]"
uv pip install -e ".[dev,train]"   # add for QLoRA / model integration
```

## Usage

```bash
# Download and normalize datasets
python scripts/download.py --output-dir data

# Train the balanced mixed Akan tokenizer
python scripts/train_bpe.py --inputs data/aka_asr_train.jsonl data/pristine_twi_train.jsonl \
    --output models/mixed_tokenizer.json --name mixed --balance

# Fine-tune a base LLM with the Akan tokenizer (QLoRA)
python scripts/model_integration.py --model-id Qwen/Qwen3-0.6B-Base
```

See [`docs/project.md`](docs/project.md) for full script options, flag reference, and experiment design.

## Revision Audit

The reviewer-revision protocol and preserved artifact hashes live in
[`config/revision_manifest.yaml`](config/revision_manifest.yaml). Validate the frozen datasets,
tokenizers, notebooks, and result artifacts without modifying them:

```bash
python scripts/validate_revision_manifest.py
```

The manifest records unavailable historical metadata explicitly and audits the one known
train/test text overlap in the historical ASR split. Revision v2 removes exactly that historical
test row, freezes the active ASR test at 1,010 rows, and rejects every active leakage exception.
New or unacknowledged overlap fails validation.

The tokenizer-only vocabulary ablation is also complete. The fixed two-domain 1% plateau rule
selects 32K for new extension and multi-seed experiments, although this is a boundary selection
rather than evidence that the curve has fully plateaued. See
[`results/vocab_ablation_tradeoff.md`](results/vocab_ablation_tradeoff.md) and
[`results/vocab_ablation_fertility.svg`](results/vocab_ablation_fertility.svg).

## Notebooks

| Notebook | nbviewer | Colab | Kaggle |
|----------|----------|-------|--------|
| Light (Qwen3-0.6B, Llama-3.2-1B, tiny-aya) | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/attabeezy/akan-bpe/blob/main/notebooks/run-full-light.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/attabeezy/akan-bpe/blob/main/notebooks/run-full-light.ipynb) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/attabeezy/akan-bpe/blob/main/notebooks/run-full-light.ipynb) |
| Heavy (Qwen3-1.7B, Gemma-3-1B) | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/attabeezy/akan-bpe/blob/main/notebooks/run-full-heavy.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/attabeezy/akan-bpe/blob/main/notebooks/run-full-heavy.ipynb) | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/attabeezy/akan-bpe/blob/main/notebooks/run-full-heavy.ipynb) |

## Project Structure

```text
akan-bpe/
├── akan_bpe/    # core library (tokenizers, metrics, router, model integration)
├── scripts/     # download, train_bpe, benchmark_fertility, router, model_integration
├── notebooks/   # ladder runs + train_eval walkthrough
├── results/     # JSON experiment outputs
└── tests/
```

## Roadmap

- [x] Akan ASR / TTS / mixed tokenizers + fertility benchmark vs XLM-R, mBERT, mT5
- [x] Heuristic + ML-classifier domain router (held-out eval)
- [x] Model-integration ladder — 5 QLoRA runs across 4 families on Kaggle/T4
- [x] Bits-per-byte metric with full byte coverage + mean-of-subword embedding-init ablation
- [x] Re-score the ladder under the corrected BPB metric
- [x] Generation quality (chrF on held-out Twi)
- [x] Revision artifact manifest and integrity validator
- [x] Leak-free ASR revision-v2 split and v1-v2 scientific checkpoint
- [x] Balanced 4K/8K/16K/32K vocabulary-size ablation
- [ ] IEEE Ghana ICAST 2026 write-up

## License

[MIT](LICENSE).
