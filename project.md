# SOMAX — Project Reference
**Eliminating the Tokenization Tax for Twi via Dual-Stream Tokenizer Experiments**

**Status:** Phase 1, tokenizer-only  
**Scope:** Twi only  
**Current hardware target:** CPU or Colab for tokenizer training and fertility benchmarking  
**Future hardware target:** Dell Latitude 7400 for downstream deployment experiments

---

## 1. Vision

SOMAX is a Twi-focused research project investigating the "Tokenization Tax":
the tendency for African languages to require far more tokens than English under
standard LLM tokenizers, increasing latency, cost, and fragmentation.

The current project is intentionally narrow.

SOMAX is not yet a model-training or deployment project. The current phase only asks:

- can specialized Twi tokenizers outperform a baseline tokenizer?
- does ASR-style Twi benefit from a different vocabulary than formal Twi?
- is one mixed tokenizer enough, or do two specialized tokenizers appear justified?

---

## 2. Current Scope

The active scope is tokenizer work only.

Included in scope:

- Twi data collection and normalization
- BPE tokenizer training
- tokenizer comparison across ASR and formal text
- token fertility benchmarking

Explicitly out of scope for now:

- model fine-tuning
- LoRA training
- embedding resizing
- GGUF export
- edge inference benchmarking
- production routing and mux deployment

These remain future directions, not current deliverables.

---

## 3. Core Idea

Twi appears to contain at least two useful text regimes:

1. **ASR / spontaneous Twi**
   This is noisy, conversational, and often includes fillers, short forms, and code-switching.

2. **Formal / TTS-like Twi**
   This is cleaner, more structured, and more semantically dense.

The main hypothesis is simple:

- a tokenizer trained on ASR-style Twi may tokenize ASR-like input more efficiently
- a tokenizer trained on formal Twi may tokenize formal input more efficiently

Before building routers or model paths, SOMAX first needs to verify that this specialization is real.

---

## 4. Research Question

The current phase asks:

**Do specialized Twi tokenizers show measurable advantages over a standard baseline tokenizer, and over each other, on different Twi text regimes?**

More concretely:

- does an ASR-trained tokenizer reduce fertility on ASR test text?
- does a TTS-trained tokenizer reduce fertility on formal test text?
- does a mixed tokenizer perform well enough that two specialized tokenizers are unnecessary?

---

## 5. Data Sources

SOMAX uses two Twi datasets:

### 5.1 WAXAL `aka_asr`

- Source: `google/WaxalNLP`
- Type: spontaneous Twi/Akan ASR transcriptions
- Characteristics:
  - conversational
  - noisy
  - filler-heavy
  - code-switching tolerant

### 5.2 Pristine-Twi

- Source: Ghana NLP `pristine-twi`
- Type: clean formal Twi text
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
| Variant A | ? | ? | Expected strength on ASR-style Twi |
| Variant B | ? | ? | Expected strength on formal Twi |
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

### Step 1: Download and normalize Twi data

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
somax/
├── data/                        # normalized Twi datasets
├── models/                      # trained tokenizer artifacts
├── results/                     # benchmark outputs
├── scripts/
│   ├── download.py              # dataset download and normalization
│   ├── train_bpe.py             # tokenizer training
│   └── benchmark_fertility.py
├── somax/                       # thin helpers for tokenizer-only experiments
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

SOMAX should use one simple rule:

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

- better Twi reasoning by a model
- better generation quality
- better LoRA adaptation
- better edge deployment performance

Those are important, but they belong to later phases.

The only thing phase 1 must prove is whether specialized tokenizers for Twi are worth pursuing.

---

## 14. Future Directions

If phase 1 shows strong specialization effects, SOMAX can expand in carefully staged steps.

### 14.1 Router / mux experiment

If Variant A and Variant B each win on their own text regime, the next logical step is:

- train a router to classify incoming Twi text as ASR-like or formal
- route the input to the most appropriate tokenizer

This would test whether a dual-tokenizer system is better than always using one tokenizer.

### 14.2 Incremental tokenizer variants

If basic A/B/C results are promising, the project can revisit staged corpus ideas such as:

- `TTS -> ASR -> TTS`
- `ASR -> TTS`

These should only be attempted after the simpler comparisons are complete.

### 14.3 Model integration

If specialized tokenizers clearly help, a later phase may explore:

- resizing model vocabularies
- initializing embeddings for new tokenizer vocabularies
- comparing specialized model paths

This is a separate project phase and should not be merged into the current tokenizer-only work.

### 14.4 Edge deployment

If tokenizer and routing experiments succeed, future work may include:

- exporting model artifacts for local inference
- benchmarking on the Dell Latitude 7400
- measuring latency, tokens per second, and memory use

### 14.5 Twi task evaluation

A later evaluation phase may test whether tokenizer gains translate to useful model behavior on tasks such as:

- Twi QA
- instruction following
- curated prompt-response evaluation

This should only happen after the tokenizer question is clearly answered.

---

## 15. Recommended Near-Term Deliverable

A successful near-term SOMAX deliverable is:

1. normalized Twi ASR and TTS datasets
2. three trained tokenizer variants: ASR, TTS, mixed
3. baseline comparison against a standard pretrained tokenizer
4. one unified experiment JSON containing the fertility comparison across ASR and TTS test sets
5. one short conclusion about whether specialization appears real

That is enough for a strong phase-1 outcome.
