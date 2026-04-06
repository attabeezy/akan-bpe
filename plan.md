# Project WAXAL-Dual-Core
**Subtitle:** Eliminating the Tokenization Tax for African Languages via Dual-Stream Tokenization

## 1. Vision
WAXAL-Dual-Core is a research-to-production framework designed to eliminate the **“Tokenization Tax”**—the structural inefficiency where African languages require significantly more tokens than English, leading to higher latency, increased cost, and degraded reasoning. 

By leveraging the **Google WAXAL dataset (Feb 2026)**, this project introduces a dual-stream tokenization architecture that treats spontaneous speech and formal text as fundamentally different linguistic regimes.

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
We will evaluate six training regimes to identify the optimal balance of robustness and efficiency:

| Group | Training Sequence | Rationale |
| :--- | :--- | :--- |
| **Control** | Standard Llama-3.2 | Baseline "Taxed" performance. |
| **Variant A** | ASR Only | Pure robustness to conversational noise. |
| **Variant B** | TTS Only | Maximum semantic density and logic. |
| **Variant C** | ASR + TTS (Mixed) | Standard joint-distribution training. |
| **Variant E** | **TTS → ASR → TTS** | **Primary Hypothesis:** Anchor logic, adapt to noise, refine logic. |
| **Variant G** | ASR → TTS | Test if phonetic grounding aids later reasoning. |

## 5. Engineering & Hardware Strategy
* **Training:** Google Colab (T4 GPU) for LoRA embedding alignment.
* **Deployment:** Dell Latitude 7400 (8GB RAM) using 4-bit GGUF quantization.
* **Product:** `WAXAL-Refined`—An open-source Python library featuring a **Dynamic Token Switcher** that routes input to the appropriate core.

## 6. Roadmap
* **Phase I (Audit):** Measure baseline Token Tax across WAXAL subsets.
* **Phase II (Train):** Staged LoRA training for Variant E and others.
* **Phase III (Patch):** Align tokens and export to GGUF format.
* **Phase IV (Release):** GitHub launch and hardware benchmarking in Ghana.