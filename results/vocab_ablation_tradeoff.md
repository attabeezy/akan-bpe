# Vocabulary-Size Ablation

| Vocab | ASR fertility | ASR p95 | ASR util. | Formal fertility | Formal p95 | Formal util. | Tokenizer MiB | Qwen 0.6B interface MiB (FP16) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4,000 | 1.384789 | 71.0 | 47.15% | 1.322836 | 491.0 | 93.05% | 0.244 | 15.62 |
| 8,000 | 1.297091 | 66.0 | 33.99% | 1.268425 | 473.0 | 86.26% | 0.502 | 31.25 |
| 16,000 | 1.240619 | 63.0 | 22.18% | 1.234157 | 461.0 | 70.69% | 1.028 | 62.50 |
| 32,000 | 1.207083 | 61.0 | 12.84% | 1.215197 | 453.0 | 47.62% | 2.097 | 125.00 |

## Operating point

Selected vocabulary: **32,000**

Rule: Select the smallest vocabulary whose fertility is within 1% relative of the best observed fertility in both ASR and formal regimes.
Status: **boundary_selected_no_earlier_plateau**

Caveat: Only the largest tested vocabulary meets the fixed threshold, so this is a boundary selection rather than evidence that the curve has fully plateaued.

Lower fertility is better. Interface memory counts untied input embeddings and the output language-model head, matching the current replacement pipeline.
