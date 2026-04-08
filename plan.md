# Project somax
**Subtitle:** Eliminating the Tokenization Tax for African Languages via Dual-Stream Processing

## 1. Vision
SOMAX is a research-to-production framework designed to eliminate the **“Tokenization Tax”**—the structural inefficiency where African languages require significantly more tokens than English, leading to higher latency, increased cost, and degraded reasoning. 

By leveraging the **Google WAXAL dataset (Feb 2026)**, SOMAX introduces a dual-stream processing architecture that treats spontaneous speech and formal text as fundamentally different linguistic regimes, sharing a unified vocabulary for maximum embedding efficiency.

## 2. Problem Statement: The Tokenization Tax
Modern LLM tokenizers are optimized for English-heavy corpora. For African languages like Akan, Yoruba, and Swahili:
* An English sentence (~10 tokens) can become 40+ tokens in the target language.
* **Consequences:** Higher Latency (edge devices), Higher Costs, and Weaker Reasoning (semantic fragmentation).
* **Metric:** Token Fertility ($F = \text{Tokens} / \text{Words}$).
* **Goal:** Reduce $F$ by $\ge 30\%$ through dual-stream vocabulary redesign.

## 3. Core Insight: Linguistic Duality
The WAXAL dataset contains two fundamentally different distributions:
1.  **WAXAL-ASR (Spontaneous):** Noisy, code-switching (e.g., Twi + English), fillers ("uhm", "chale"), and disfluencies.
2.  **WAXAL-TTS (Formal):** Clean, structured, grammatically correct, and semantically dense scripts.

## 4. Methodology: Experimental Groups
We evaluate six training regimes to identify the optimal balance of robustness and efficiency:

| Group | Training Sequence | Rationale |
| :--- | :--- | :--- |
| **Control** | Standard Llama-3.2 | Baseline "Taxed" performance. |
| **Variant A** | ASR Only | Pure robustness to conversational noise. |
| **Variant B** | TTS Only | Maximum semantic density and logic. |
| **Variant C** | ASR + TTS (Mixed) | Standard joint-distribution training. |
| **Variant D** | **TTS → ASR → TTS** | **Primary Hypothesis:** Anchor logic, adapt to noise, refine logic. |
| **Variant E** | ASR → TTS | Test if phonetic grounding aids later reasoning. |

## 5. Engineering & Hardware Strategy
* **Training:** Google Colab (T4 GPU) for staged LoRA embedding alignment.
* **Deployment:** Dell Latitude 7400 (8GB RAM) using 4-bit GGUF quantization.
* **Product:** `somax`—An open-source Python library featuring a **Dynamic Stream Router** and a **Dual-Core Tokenizer** sharing a unified BPE vocabulary.

## 6. Roadmap
* **Phase I (Audit):** Measure baseline Token Tax across WAXAL subsets using `benchmark_fertility.py`.
* **Phase II (Train):** Staged LoRA training for Variant D and others via `train_lora.py`.
* **Phase III (Patch):** Export to GGUF format for edge deployment using `export_gguf.py`.
* **Phase IV (Release):** Hardware benchmarking on the Dell Latitude 7400 and GitHub launch.