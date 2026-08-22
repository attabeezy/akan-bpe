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

The Qwen 0.6B extension contract is now frozen as well: 26,156 novel tokens from that 32K
candidate vocabulary are appended without changing any original token ID. Its intrinsic v2
comparison and controlled BPB/chrF runs are complete; downstream task evaluation remains pending.

### AfriSenti downstream evaluation

The frozen P1 downstream protocol evaluates the Twi AfriSenti benchmark with two original
Qwen bases and nine recreated adapted checkpoints (three strategies/model combinations by
seeds 17, 42, and 73). It reports the official 949-row test split and a 730-row sensitivity
surface that excludes normalized train/validation duplicates.

```bash
python scripts/run_downstream_afrisenti.py validate
python scripts/run_downstream_afrisenti.py fetch-data
python scripts/run_downstream_afrisenti.py status
python scripts/run_downstream_afrisenti.py run --next
python scripts/run_downstream_afrisenti.py aggregate
```

Model runs require a CUDA GPU. `notebooks/downstream_afrisenti.ipynb` runs resumable batches
on a single Kaggle T4 and deletes each temporary checkpoint only after its result validates.

Routing is now explicitly secondary analysis. The frozen audit reconstructs the historical
TF-IDF/logistic-regression protocol and metrics, but the near-perfect result separates two source
corpora rather than a genuinely ambiguous domain challenge set. The balanced mixed tokenizer is
the primary deployment path; robust routing is deferred.

The remaining P0 GPU evidence is execution-ready through one frozen 15-run contract:

```bash
python scripts/run_revision_gpu_matrix.py validate
python scripts/run_revision_gpu_matrix.py status
python scripts/run_revision_gpu_matrix.py run --next
python scripts/aggregate_revision_gpu_matrix.py
```

Run one `--next` job per fresh Kaggle/Colab GPU process. Results are resumable and the aggregator
will reject incomplete or configuration-drifted matrices.

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
- [x] Qwen vocabulary-extension builder and intrinsic checkpoint
- [x] Router protocol audit and demotion to secondary analysis
- [x] Frozen extension/replacement and multi-seed GPU execution harness
- [ ] IEEE Ghana ICAST 2026 write-up

## License

[MIT](LICENSE).
