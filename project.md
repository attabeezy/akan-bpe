# Akan-BPE — Project Reference
**Eliminating the Tokenization Tax for Akan via BPE Tokenizer Experiments**

**Status:** Phase 2A In Progress (v0.2.0) — 2A1 complete (first Colab QLoRA run); 2A2 next  
**Scope:** Akan (Twi), tokenizer experiments with ML routing  
**Completed:** Tokenizer training, fertility benchmarks vs multilingual baselines, balanced mixed tokenizer, router with held-out eval, first model-integration run (Qwen3-0.6B + Akan TTS tokenizer on Colab/T4)  
**Current hardware:** CPU (local) / Colab T4 (model integration)  
**Next hardware:** Continued Colab/Kaggle GPU for the model ladder; Dell Latitude 7400 for edge deployment

---

## 1. Vision

Akan-BPE is an Akan-focused research project investigating the "Tokenization Tax":
the tendency for African languages to require far more tokens than English under
standard LLM tokenizers, increasing latency, cost, and fragmentation.

The current project is intentionally narrow.

Akan-BPE is not yet a model-training or deployment project. The current phase only asks:

- can specialized Akan tokenizers outperform a baseline tokenizer?
- does ASR-style Akan benefit from a different vocabulary than formal Akan?
- is one mixed tokenizer enough, or do two specialized tokenizers appear justified?

---

## 2. Current Scope

The active scope is tokenizer + routing experiments.

**Completed:**
- Akan data collection and normalization (80/10/10 local split for ASR)
- BPE tokenizer training (ASR, TTS, Mixed)
- Tokenizer comparison against multilingual baselines (XLM-R, mBERT, mT5)
- Token fertility benchmarking (~52% ASR reduction, ~47% TTS reduction vs best baseline)
- Balanced mixed tokenizer (corpus upsampling — now genuinely differentiates domains)
- Heuristic router implementation
- ML classifier router (99.99% train/test accuracy on stratified held-out split)

**In progress / next phases:**
- Model integration — 2A1 complete: Qwen3-0.6B QLoRA fine-tune with the Akan TTS tokenizer on Colab/T4 (49.5% fertility reduction, eval perplexity 83.81, coherent Twi generation). 2A2 (Qwen3-1.7B) is next.
- Edge deployment (GGUF export, Dell Latitude 7400 benchmarking) — blocked on a usable Phase 2A model artifact

---

## 3. Core Idea

Akan appears to contain at least two useful text regimes:

1. **ASR / spontaneous Akan**
   This is noisy, conversational, and often includes fillers, short forms, and code-switching.

2. **Formal / TTS-like Akan**
   This is cleaner, more structured, and more semantically dense.

The main hypothesis is simple:

- a tokenizer trained on ASR-style Akan may tokenize ASR-like input more efficiently
- a tokenizer trained on formal Akan may tokenize formal input more efficiently

Before building routers or model paths, Akan-BPE first needs to verify that this specialization is real.

---

## 4. Research Question

The current phase asks:

**Do specialized Akan tokenizers show measurable advantages over a standard baseline tokenizer, and over each other, on different Akan text regimes?**

More concretely:

- does an ASR-trained tokenizer reduce fertility on ASR test text?
- does a TTS-trained tokenizer reduce fertility on formal test text?
- does a mixed tokenizer perform well enough that two specialized tokenizers are unnecessary?

---

## 5. Data Sources

Akan-BPE uses two Akan datasets:

### 5.1 WAXAL `aka_asr`

- Source: `google/WaxalNLP`
- Type: spontaneous Akan ASR transcriptions
- Characteristics:
  - conversational
  - noisy
  - filler-heavy
  - code-switching tolerant

### 5.2 Pristine-Twi

- Source: Ghana NLP `pristine-twi`
- Type: clean formal Akan text
- Characteristics:
  - structured
  - grammatically cleaner
  - more formal and semantically dense

These two corpora define the dual-stream tokenizer experiment.

---

## 6. Phase 1 Experimental Design

This phase compares tokenizers only.

### 6.1 Tokenizer Variants

The recommended tokenizer variants are:

| Variant | Description | Purpose |
|---|---|---|
| **Control** | Existing baseline tokenizer from a pretrained model | Reference point |
| **Variant A** | Tokenizer trained only on ASR text | Specialized conversational tokenizer |
| **Variant B** | Tokenizer trained only on formal/TTS text | Specialized formal tokenizer |
| **Variant C** | Tokenizer trained on mixed ASR + TTS text | Single-tokenizer compromise |

For now, these are tokenizer variants, not model variants.

### 6.2 Deferred Variants

The original project also considered staged variants such as:

- `TTS -> ASR -> TTS`
- `ASR -> TTS`

Those ideas are not the first priority in tokenizer-only phase 1.
They may be revisited later if the basic A/B/C results show clear separation.

---

## 7. Experimental Goal

The immediate goal is to produce one clean comparison table across two test sets.

Target benchmark table:

| Tokenizer | ASR Test Fertility | TTS Test Fertility | Interpretation |
|---|---:|---:|---|
| Control | baseline | baseline | Standard reference |
| Variant A | ? | ? | Expected strength on ASR-style Akan |
| Variant B | ? | ? | Expected strength on formal Akan |
| Variant C | ? | ? | Mixed compromise candidate |

This table is the primary deliverable for phase 1.

---

## 8. Metric

### Primary metric: Token Fertility

Token fertility is defined as:

`F = total_tokens / total_words`

This is the main evaluation metric for the current phase.

Interpretation:

- lower is better, if text quality and meaning preservation are not being altered
- a tokenizer is more efficient when it needs fewer tokens per word on the same text

### Phase 1 success criteria

Success in phase 1 does not require a complete product.
It requires a clear empirical result, such as:

- Variant A performs best on ASR test text
- Variant B performs best on TTS test text
- Variant C performs competitively on both
- or one tokenizer dominates both regimes and weakens the dual-tokenizer hypothesis

Any of those are valid findings.

---

## 9. Recommended Workflow

The current recommended workflow is:

### Step 1: Download and normalize Akan data

Use `download.py` to create standardized JSONL files under `data/`.

Recommended filenames:

- `aka_asr_train.jsonl`
- `aka_asr_validation.jsonl`
- `aka_asr_test.jsonl`
- `pristine_twi_train.jsonl`
- `pristine_twi_validation.jsonl`
- `pristine_twi_test.jsonl`

### Step 2: Train tokenizer variants

Train:

- ASR tokenizer from `aka_asr_train.jsonl`
- TTS tokenizer from `pristine_twi_train.jsonl`
- mixed tokenizer from both training sets

All tokenizer variants should use:

- the same algorithm
- the same vocab size
- the same special tokens

This keeps the comparison fair.

### Step 3: Benchmark fertility

Run one unified benchmark experiment that evaluates all selected tokenizers on:

- ASR test text
- TTS test text

This should produce one comparison JSON, not many small result files.

### Step 4: Interpret the results

Possible outcomes:

- specialization is real
- one mixed tokenizer is enough
- one tokenizer dominates everything

Only after that should the project consider routing or model work.

---

## 10. Repository Structure

The current project should be understood through this simplified structure:

```text
akan_bpe/
├── data/                        # normalized Akan datasets
├── models/                      # trained tokenizer artifacts
├── results/                     # benchmark outputs
├── scripts/
│   ├── download.py              # dataset download and normalization
│   ├── train_bpe.py             # tokenizer training
│   └── benchmark_fertility.py
├── akan_bpe/                       # thin helpers for tokenizer-only experiments
├── tests/
├── README.md
└── project.md
```

---

## 11. Canonical File Contracts

### 11.1 Data files

Recommended JSONL schema:

```json
{"id": "sample_id", "text": "some twi text", "source": "aka_asr"}
```

If existing scripts use `transcription`, that is acceptable, but the repo should converge on one field contract over time.

### 11.2 Tokenizer artifacts

Recommended outputs:

- `models/asr_tokenizer.json`
- `models/tts_tokenizer.json`
- `models/mixed_tokenizer.json`

Optional metadata:

- training stats
- corpus sizes
- vocab summaries

### 11.3 Benchmark outputs

Akan-BPE should use one simple rule:

- one experiment run produces one JSON file

Recommended result file:

- `results/tokenizer_fertility_experiment_001.json`

That file should contain:

- experiment metadata
- the tokenizers included in the run
- the test sets used
- fertility results for every tokenizer on every test set
- a short summary of which tokenizer performed best where

The project should avoid scattering one experiment across many small output files.

---

## 12. Best Practices For Phase 1

To keep the project small and defensible:

- vary one major factor at a time
- keep vocab size constant across tokenizer variants
- keep special tokens constant across tokenizer variants
- use the same test files for every benchmark
- save every benchmark result to JSON
- treat one benchmark run as one complete experiment with one output JSON
- avoid mixing tokenizer experiments with model experiments
- document the exact corpus used for each tokenizer

This phase should produce a clear result before the repo takes on more complexity.

---

## 13. What This Phase Is Not Trying To Prove

Phase 1 is not trying to prove:

- better Akan reasoning by a model
- better generation quality
- better LoRA adaptation
- better edge deployment performance

Those are important, but they belong to later phases.

The only thing phase 1 must prove is whether specialized tokenizers for Akan are worth pursuing.

---

## 14. Future Directions

If phase 1 shows strong specialization effects, Akan-BPE can expand in carefully staged steps.

### 14.1 Router / mux experiment (COMPLETED)

- Implemented heuristic-based router (77.6% accuracy)
- Trained ML classifier (TF-IDF + Logistic Regression, 99.99% train/test accuracy on stratified 80/20 split)
- Per-class F1: ASR 0.9998, TTS 0.9999
- Benchmark showed ML router achieves optimal fertility (matches always-best-tokenizer strategy)

**Status:** Complete - ML router significantly outperforms heuristic; accuracy confirmed on held-out test set

### 14.2 Incremental tokenizer variants

If basic A/B/C results are promising, the project can revisit staged corpus ideas such as:

- `TTS -> ASR -> TTS`
- `ASR -> TTS`

These should only be attempted after the simpler comparisons are complete.

### 14.3 Model integration (IN PROGRESS — Phase 2A)

Phase 2A1 is complete. The first real Colab QLoRA run executed end-to-end. The repo contains:

- `akan_bpe/model_integration.py` — dataset prep, tokenizer/model loading, token-count comparison, LoRA/QLoRA setup, eval, generation samples, JSON artifact creation
- `scripts/model_integration.py` — CLI-driven experiment runner
- `phase2a_qwen3_tts_colab.ipynb` — executed Colab/T4 run for `Qwen/Qwen3-0.6B` (outputs preserved in the notebook)
- `tests/test_model_integration.py` — CPU-safe orchestration and artifact-contract coverage

**2A1 result (Qwen3-0.6B + Akan TTS tokenizer, QLoRA 4-bit nf4, Tesla T4, 1 epoch):**

- Base model tokenizer fertility: 2.538 tokens/word → Akan TTS tokenizer: 1.280 tokens/word (**49.5% reduction** on the eval set)
- Eval loss 4.4285 / perplexity 83.81
- Generation produces coherent Twi continuations; save → reload-from-adapter inference verified

Next step is 2A2 (`Qwen/Qwen3-1.7B`).

### 14.4 Edge deployment

If tokenizer and routing experiments succeed, future work may include:

- exporting model artifacts for local inference
- benchmarking on the Dell Latitude 7400
- measuring latency, tokens per second, and memory use

### 14.5 Akan task evaluation

A later evaluation phase may test whether tokenizer gains translate to useful model behavior on tasks such as:

- Akan QA
- instruction following
- curated prompt-response evaluation

This should only happen after the tokenizer question is clearly answered.

---

## 15. Phase 1 Deliverables (COMPLETE)

1. ✅ normalized Akan ASR and TTS datasets (80/10/10 local split)
2. ✅ three trained tokenizer variants: ASR, TTS, Mixed (corpus-balanced)
3. ✅ fertility benchmark vs multilingual baselines (XLM-R, mBERT, mT5) — not GPT-2
4. ✅ unified experiment JSON with fertility comparison
5. ✅ technical report (report.md) documenting findings
6. ✅ ML classifier router (99.99% train/test accuracy, stratified held-out eval)
7. ✅ End-to-end notebook (train_eval.ipynb)

**Conclusion:** Specialization is real — ASR tokenizer achieves ~52% fertility reduction, TTS ~47%, both vs best multilingual baseline (mBERT). Balanced mixed tokenizer interpolates between domains (1.20 ASR, 1.27 TTS) and is viable where routing infrastructure is unavailable.

---

## 16. Phase 2: Next Steps

Phase 1 answered the tokenizer question. Phase 2 asks whether those gains translate to a real model.

### 16.1 Model Integration

**Goal:** Verify that fertility reduction translates into measurable downstream benefit — faster inference, lower perplexity, or better generation — not just a smaller token count.

**Current status:** 2A1 is complete — the first real Colab/T4 QLoRA run executed end-to-end on `Qwen/Qwen3-0.6B` with the Akan TTS tokenizer (49.5% fertility reduction on the eval set, eval perplexity 83.81, coherent Twi generation, save/reload verified). The repo contains:

- `akan_bpe/model_integration.py` for dataset prep, tokenizer/model loading, token-count comparison, LoRA/QLoRA setup, eval, generation samples, and JSON artifact creation
- `scripts/model_integration.py` for one CLI-driven experiment run
- `phase2a_qwen3_tts_colab.ipynb` — the executed Colab run (outputs preserved in-notebook; `results/` is gitignored)
- `tests/test_model_integration.py` for CPU-safe orchestration and artifact-contract coverage

**Hardware baseline:** Free Kaggle/Colab GPU, typically T4/P100-class. Train and evaluate the smaller models first; treat larger models as QLoRA-only or reference-only unless paid GPU access is available.

**Model ladder:**

| Phase | Model | Role | Free-GPU feasibility |
|---|---|---|---|
| **2A1 ✅** | `Qwen/Qwen3-0.6B` | First real Colab/T4 QLoRA run: tokenizer replacement, embedding resize, train/eval, generation, save/load verification — **done** (49.5% fertility reduction, perplexity 83.81) | Feasible — completed |
| **2A2** | `Qwen/Qwen3-1.7B` | Main small-model experiment after 2A1 proves the path | Feasible with LoRA/QLoRA |
| **2A3** | `google/gemma-3-1b-*` | Broad multilingual vendor/architecture comparison | Feasible; check Gemma license and PT/IT choice |
| **2A4** | `meta-llama/Llama-3.2-1B` or `meta-llama/Llama-3.2-3B` | Deployment ecosystem comparison | 1B feasible; 3B QLoRA with conservative settings |
| **2A5** | `CohereLabs/tiny-aya-earth` | Africa/West Asia-focused multilingual experiment | QLoRA-only; CC-BY-NC research/non-commercial license |
| **2A6** | `microsoft/Phi-4-mini-instruct` or `CohereLabs/aya-expanse-8b` | Stretch/reference tier: Phi for edge-quality upper bound, Aya Expanse for multilingual reference | Phi QLoRA stretch; Aya Expanse inference/reference-only |

`Qwen2.5-0.5B` remains a fallback only if Qwen3 tooling causes friction. Tiny Aya Earth replaces Aya Expanse 8B as the primary Aya-family candidate because it is smaller, Africa/West Asia-focused, and designed for local deployment under realistic compute constraints.

**Steps:**

1. **Choose tokenizer to integrate** — start with the TTS tokenizer (most training data, best-in-class fertility on formal text). The router can be layered in later.

2. **Resize token embeddings**
   ```python
   from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

   base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
   new_tokenizer = PreTrainedTokenizerFast(tokenizer_file="models/tts_tokenizer.json")
   base_model.resize_token_embeddings(len(new_tokenizer))
   ```
   New token embeddings initialize randomly; existing tokens that map to the new vocab keep their weights where possible.

3. **Re-tokenize training data** — run `pristine_twi_train.jsonl` through the new tokenizer to produce training inputs for fine-tuning.

4. **Fine-tune** — LoRA is the practical choice on limited hardware. Full fine-tune only if GPU VRAM allows.
   - Library: `peft` + `transformers` Trainer or `trl` SFTTrainer
   - Target modules: attention Q/K/V projections
   - Rank: r=8 or r=16 to start

5. **Evaluate**
   - **Perplexity** on `pristine_twi_test.jsonl` — compare base model (original tokenizer) vs fine-tuned (new tokenizer)
   - **Generation quality** — BLEU or chrF on a small Akan reference set if available; otherwise qualitative review
   - **Inference speed** — tokens/second before and after to quantify the fertility gain in practice

6. **Record experiment output** — save one structured JSON per run under `results/`, including model ID, tokenizer path, dataset paths, fertility, perplexity, generation samples, timing, hardware, and memory notes.

**First real run path:** Use `phase2a_qwen3_tts_colab.ipynb` to install `.[dev,train]` plus `bitsandbytes`, verify `data/pristine_twi_train.jsonl`, `data/pristine_twi_test.jsonl`, and `models/tts_tokenizer.json`, then call `scripts/model_integration.py` in `colab-qlora` mode.

**Success criterion:** Fine-tuned model with new tokenizer matches or exceeds base model perplexity on Akan test text, with fewer tokens processed per sample.

**Failure mode to watch for:** If perplexity is significantly worse after embedding resize, the initialization strategy needs work (e.g., averaging subword embeddings from the original vocab that cover similar character sequences).

#### 16.1.1 Phase 2A2 — next actions (`Qwen/Qwen3-1.7B`)

The 2A1 path is proven, so 2A2 is mostly a configuration change on the same machinery.
Concrete steps:

1. **Extend the QLoRA allowlist.** `colab-qlora` mode is intentionally pinned to one
   model: `SUPPORTED_COLAB_QLORA_MODEL_IDS = ("Qwen/Qwen3-0.6B",)` in
   `akan_bpe/model_integration.py`. Add `"Qwen/Qwen3-1.7B"` there (and update
   `validate_colab_qlora_config` coverage in `tests/test_model_integration.py`).
2. **Clone the notebook.** Copy `phase2a_qwen3_tts_colab.ipynb` to a 2A2 variant and
   point `--model-id` at `Qwen/Qwen3-1.7B`; keep the TTS tokenizer and dataset paths.
   1.7B in 4-bit fits a free T4 but may need a smaller `--batch-size` / higher
   `--grad-accum` than 2A1.
3. **Compare against 2A1.** Record the same metrics (fertility reduction, eval
   perplexity, generation quality) so 2A1 vs 2A2 is an apples-to-apples ladder step.

**Known limitation carried into Phase 2:** the ASR test split is effectively a single
sample (see §5 / `report.md` §2.1), so any ASR-side evaluation is anecdotal. If ASR
becomes a model-integration target (vs the current TTS focus), regenerate a real ASR
test split first via `scripts/download.py` (the `_split_rows` 80/10/10 logic already
exists; the current tiny split is a stale-download artifact, not a code bug).

---

### 16.2 Edge Deployment

**Goal:** Benchmark tokenizer + router + model on the Dell Latitude 7400 to understand real-world latency and memory footprint.

**Prerequisite:** Model integration (16.1) must produce a usable model artifact first.

**Steps:**

1. **Export to GGUF**
   ```bash
   python llama.cpp/convert_hf_to_gguf.py models/akan_tts_model/ --outtype q4_k_m
   ```
   Q4_K_M quantization is a good starting point — balances quality and size for a 0.5–1B model.

2. **Bundle the router** — the router classifier (`models/router_classifier.pkl`) adds ~1ms per call; confirm this overhead is negligible on the target hardware.

3. **Benchmark on Dell Latitude 7400**

   Metrics to collect:
   | Metric | Tool |
   |--------|------|
   | Tokens/second | `llama-bench` or manual timing |
   | Peak RAM | `psutil` or Task Manager |
   | Time-to-first-token | manual timing in Python |
   | Router overhead | `time.perf_counter()` around `classifier.predict()` |

4. **Compare configurations**
   - Base model, original tokenizer (GPT-2 vocab)
   - Fine-tuned model, Akan TTS tokenizer
   - Fine-tuned model + ML router (dynamic tokenizer selection)

**Success criterion:** Fine-tuned Akan model generates tokens faster (more tokens/second or fewer tokens per prompt) than the base model on Akan input, with acceptable RAM footprint for the target hardware.

---

### 16.3 Sequencing

```
Phase 1 (DONE)
    └── Phase 2A: Model Integration
            ├── 2A1 Qwen3-0.6B first Colab/T4 QLoRA run (DONE)
            ├── 2A2 Qwen3-1.7B main small-model experiment (NEXT)
            ├── 2A3 Gemma 3 1B multilingual comparison
            ├── 2A4 Llama 3.2 1B/3B deployment comparison
            ├── 2A5 Tiny Aya Earth Africa-focused QLoRA experiment
            ├── 2A6 Phi/Aya Expanse stretch/reference tier
            ├── Resize + LoRA/QLoRA fine-tune on TTS data
            ├── Eval fertility, perplexity, generation quality, timing
            └── Phase 2B: Edge Deployment
                    ├── GGUF export
                    ├── Bundle router
                    └── Benchmark on Dell Latitude 7400
```

Phase 2B is blocked on Phase 2A. Do not start edge deployment until there is a working fine-tuned model artifact to benchmark.
