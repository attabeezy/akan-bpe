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
- [ ] IEEE Ghana ICAST 2026 write-up

## License

[MIT](LICENSE).
