# somax

**Eliminating the Tokenization Tax for African Languages via Dual-Stream Tokenization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

Modern LLM tokenizers are optimized for English, resulting in a **"Tokenization Tax"** for African languages like Akan, Yoruba, and Swahili. This project implements a research-to-production framework that:

- Reduces token fertility (F = tokens/words) by ≥30%
- Uses dual-stream processing for spontaneous speech (ASR) and formal text (TTS)
- Deploys efficiently on edge devices (8GB RAM)

## Key Components

### 1. **WAXALRouter** (`somax/router.py`)
Lightweight heuristic classifier that routes input to the appropriate stream:
- **Robust stream** (ASR-optimized): Handles conversational text, fillers, code-switching
- **Logic stream** (TTS-optimized): Handles formal, semantic-rich text

### 2. **DualCoreTokenizer** (`somax/tokenizer.py`)
Manages stream-aware tokenization using a **Unified Vocabulary**. Both streams share a single BPE vocabulary to ensure embedding alignment, while the router identifies the linguistic regime for downstream model selection.

### 3. **Training Pipeline** (`scripts/`)
- `download.py` - Download WAXAL dataset from HuggingFace
- `train_bpe.py` - Train unified BPE vocabulary for ASR/TTS
- `train_router.py` - Train the TF-IDF router classifier
- `train_lora.py` - Train LoRA variants (A, B, C, D, E)
- `export_gguf.py` - Export to GGUF for edge deployment

## Quick Start

### Prerequisites

```bash
# Core dependencies
pip install transformers peft bitsandbytes datasets accelerate
pip install torch sentencepiece tokenizers

# For edge deployment
pip install llama-cpp-python psutil

# For development
pip install -e ".[dev,train]"
```

### Option 1: Run End-to-End in Google Colab

1. Open `notebooks/pipeline.ipynb` in VS Code
2. Install the Colab extension
3. Set your HuggingFace token: `os.environ['HF_TOKEN'] = 'your_token'`
4. Run all cells

### Option 2: Run Locally

```bash
# 1. Download WAXAL dataset (Akan)
python scripts/download.py --lang akan --output data/

# 2. Train unified BPE vocabulary
python scripts/train_bpe.py --input data/akan/ --output models/tokenizers/

# 3. Train LoRA variant (D recommended)
python scripts/train_lora.py --group D --data data/akan/ --output checkpoints/

# 4. Benchmark fertility results
python benchmark_fertility.py --tokenizer meta-llama/Llama-3.2-1B \
    --waxal-tokenizer models/tokenizers/akan/unified_tokenizer.json --test-file data/akan/twi_tts_test.jsonl --compare

# 5. Export to GGUF (requires llama.cpp)
python scripts/export_gguf.py --checkpoint checkpoints/variant_D/final/ --output models/gguf/

# 6. Benchmark edge inference
python benchmark_inference.py --model models/gguf/variant_D_Q4.gguf --test-file data/akan/twi_tts_test.jsonl
```

## Dataset

**WAXAL Dataset** (Google Research, Feb 2026)
- **Source**: `google/WaxalNLP` on HuggingFace
- **Languages**: Akan, Yoruba, Swahili (others available)
- **Splits**: ASR (spontaneous) and TTS (formal)
- **License**: CC-BY-4.0

### Language Configuration

| Language | ASR Config | TTS Config | Notes |
|----------|------------|-------------|-------|
| Akan | `aka_asr` | `twi_tts` | Proof-of-concept |
| Yoruba | *(none)* | `yor_tts` | TTS only |
| Swahili | *(none)* | `swa_tts` | TTS only |

## Training Variants

| Group | Training Sequence | Rationale |
|-------|-------------------|-----------|
| **Control** | Standard Llama-3.2-1B | Baseline "Taxed" performance |
| **Variant A** | ASR Only | Pure robustness to conversational noise |
| **Variant B** | TTS Only | Maximum semantic density and logic |
| **Variant C** | ASR + TTS (Mixed) | Standard joint-distribution training |
| **Variant D** | TTS → ASR → TTS | **Primary Hypothesis**: Anchor logic, adapt to noise, refine logic |
| **Variant E** | ASR → TTS | Test if phonetic grounding aids later reasoning |

## Metrics

### Primary: Token Fertility
```
F = Total Tokens / Total Words
```
**Target**: Reduce F by ≥30% vs baseline

### Secondary
- Tokens per second (TPS)
- Inference latency (seconds)
- Memory usage (MB)

## Hardware Requirements

### Training (Cloud)
- **Platform**: Google Colab
- **GPU**: T4 (16GB VRAM)
- **Runtime**: ~2-4 hours per variant

### Deployment (Edge)
- **Device**: Dell Latitude 7400
- **RAM**: 8GB
- **Format**: 4-bit GGUF quantization

## Project Structure

```
SOMAX/
├── data/                  # WAXAL subsets (gitignored)
├── models/                # Trained models (gitignored)
├── scripts/               # Training pipeline
│   ├── download.py
│   ├── train_bpe.py
│   ├── train_router.py
│   ├── train_lora.py
│   └── export_gguf.py
├── somax/         # Edge Python library
│   ├── __init__.py
│   ├── router.py          # Stream classifier
│   └── tokenizer.py       # Dual-core tokenizer (Unified)
├── notebooks/             # Colab notebooks
│   └── pipeline.ipynb
├── benchmark_fertility.py  # Fertility metrics
├── benchmark_inference.py  # Edge performance metrics
├── requirements.txt
├── pyproject.toml
├── Makefile
└── README.md
```

## Usage Example

```python
from somax import DualCoreTokenizer

# Initialize with unified tokenizer
tokenizer = DualCoreTokenizer(
    tokenizer_path="models/tokenizers/akan/unified_tokenizer.json",
    language="akan"
)

# Automatic routing based on input
conversational = "uhm chale me dwo o"  # Routed to ASR stream
formal = "The president delivered a formal address"  # Routed to TTS stream

# Check routing
print(tokenizer.classify(conversational))  # "robust"
print(tokenizer.classify(formal))           # "logic"

# Encode
tokens = tokenizer.encode(conversational)
```

## Results

After training, compare with baseline:

```bash
# Baseline
python benchmark_fertility.py --tokenizer meta-llama/Llama-3.2-1B --test-file data/akan/twi_tts_test.jsonl

# Trained
python benchmark_fertility.py --tokenizer models/tokenizers/akan/unified_tokenizer.json --waxal --test-file data/akan/twi_tts_test.jsonl
```

Expected improvement:
- **Baseline fertility**: ~4.0 tokens/word
- **Target fertility**: ~2.8 tokens/word (30% reduction)

## Roadmap

- [x] Project skeleton
- [x] Dataset integration (WaxalNLP)
- [x] Training scripts (LoRA variants)
- [x] Edge library (router + tokenizer)
- [x] Colab notebook
- [ ] Benchmark on edge hardware (Dell Latitude 7400)
- [ ] Open-source model release

## Citation

If you use this work, please cite the WAXAL dataset:

```bibtex
@article{waxal2026,
  title={WAXAL: A Large-Scale Multilingual African Language Speech Corpus},
  author={Anonymous},
  journal={arXiv preprint arXiv:2602.02734},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Research for the WAXAL dataset
- HuggingFace for model distribution
- The African NLP community
