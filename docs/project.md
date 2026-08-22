# Akan-BPE — Project Reference
**Eliminating the Tokenization Tax for Akan via BPE Tokenizer Experiments**

**Status:** ICAST reviewer-revision preparation. The frozen 15-run revision-v2 GPU matrix is
complete and validated in `results/revision_v2/gpu_runs`, with generated statistics in
`results/revision_v2/gpu_matrix_aggregate.json`. The remaining critical experiment is the P1
downstream Akan task; qualitative analysis, manuscript revision, and final QA also remain.
**Scope:** Akan (Twi), tokenizer experiments with ML routing
**Paper target:** IEEE Ghana ICAST 2026. The active plan is now driven by
this ICAST submission — see §0 (Research Design & Road to Paper) for the locked decisions and the
milestone road. §0 takes precedence where it tightens or narrows the longer Phase 2 plan below.
**Completed:** Tokenizer training, fertility benchmarks vs multilingual baselines, balanced mixed tokenizer, router with held-out eval, the 5-run model-integration ladder (run-qwen-0.6b … run-aya-base) with the balanced mixed tokenizer on Kaggle/T4, leak-free ASR revision v2, and the 4K/8K/16K/32K vocabulary-size ablation
**Current hardware:** CPU (local) / Kaggle T4 (model integration)
**Next hardware:** Continued Colab/Kaggle GPU for the model ladder; Dell Latitude 7400 for edge deployment

---

## Reviewer Revision Plan (Authoritative)

**Source:** `docs/review_tokenization_tax.pdf`, reviewer report dated July 24, 2026.
**Objective:** Resolve every required concern, address the strongest recommendations, and
produce a reproducible revision package for the paper.
**Precedence:** This section supersedes older statements elsewhere in this file where they
conflict. In particular, single-seed evidence is no longer sufficient, a downstream task is no
longer deferred by default, and the router must be documented or demoted.

### R1. Definition of done

The revision is ready only when:

- [x] All four reviewer-required additions are complete: vocabulary-size ablation, vocabulary
  extension baseline, multi-seed runs, and router documentation/demotion.
- [ ] BPB, chrF/chrF++, tokenizer training, QLoRA, and decoding protocols are fully specified.
- [ ] At least one useful downstream Akan task has been evaluated on at least two adapted models.
- [ ] Every reported number can be regenerated from a committed script/notebook and traced to a
  structured result artifact.
- [ ] Tables and figures are regenerated from the final artifacts rather than copied manually.
- [ ] The manuscript clearly separates intrinsic tokenizer efficiency, language-model quality,
  downstream usefulness, and deployment implications.
- [ ] Claims are limited to the evidence, licenses and attribution are checked, and the final PDF
  passes a visual and reference-consistency review.

### R2. Priority and execution order

| Priority | Workstream | Dependency | Required output |
|---|---|---|---|
| **P0** | Freeze protocols and artifact contracts | None | Versioned experiment specification |
| **P0** | Vocabulary-size ablation | Frozen tokenizer protocol | 4K/8K/16K/32K results and figure |
| **P0** | Vocabulary-extension baseline | Frozen integration protocol | Extension vs replacement comparison |
| **P0** | Multi-seed validation | Stable model pipeline | Mean, standard deviation, and per-seed artifacts |
| **P0** | Router documentation or demotion | Existing router artifacts | Rewritten router section/appendix |
| **P1** | Metric and hyperparameter clarification | Frozen protocols | Exact BPB, chrF, and QLoRA descriptions |
| **P1** | Downstream task evaluation | Stable adapted checkpoints | Task metrics for at least two models |
| **P1** | Qualitative tokenization analysis | Final tokenizer set | 5-10 annotated examples |
| **P1** | Paper restructuring and related work | Final primary results | Revised manuscript |
| **P2** | Unigram/WordPiece fertility comparison | Data/tokenizer pipeline | Algorithm comparison table |
| **P2** | Longer-training ablation | Multi-seed baseline | 1-vs-3-epoch comparison |
| **P2** | Optional latency measurement | Stable model artifact | Carefully scoped deployment evidence |

Run the work in this order:

1. Freeze evaluation inputs, seeds, configuration schemas, and naming conventions.
2. Run cheap tokenizer-only experiments and qualitative analysis.
3. Implement and smoke-test vocabulary extension.
4. Run the GPU-heavy multi-seed and downstream evaluations.
5. Regenerate all aggregate artifacts, tables, and figures.
6. Rewrite the paper around the final evidence.
7. Perform reproducibility, statistical, visual, citation, and submission QA.

### R3. P0 - Freeze the experimental protocol

**Implementation status (July 30, 2026):** The preservation foundation and leak-free revision-v2
boundary are implemented in `config/revision_manifest.yaml`, with a read-only validator at
`scripts/validate_revision_manifest.py`. Historical v1 remains hash-pinned. Revision v2 removes
only historical test row 368, leaves the 8,085-row train and 1,011-row validation files unchanged,
and freezes the active test at 1,010 rows. The validator checks hashes, sizes, row counts, result
assertions, IDs, and normalized-text overlap, and passes with zero active leakage exceptions.
Affected fertility and router metrics were regenerated from v2. Their v1-v2 checkpoint found no
material change and retains the historical model ladder.

- [x] Create one revision experiment manifest recording:
  - Git commit, environment/package versions, hardware, model revision, and dataset hashes.
  - Exact train/validation/test files and row counts.
  - Tokenizer algorithm, normalization, pre-tokenization, special tokens, minimum frequency,
    vocabulary size, and random seed.
  - Model ID/revision, tokenizer strategy, initialization mode, QLoRA settings, training seed,
    decoding settings, and evaluation limits.
- [x] Keep the ASR train/validation files fixed at 8,085/1,011 and use the documented,
  leak-corrected 1,010-row revision-v2 test; keep the formal split at 45,000/2,500/2,500.
- [x] Prevent train/evaluation leakage with explicit ID/hash overlap checks.
- [x] Adopt stable result IDs that encode model, tokenizer strategy, vocabulary size,
  initialization, epoch count, and seed.
- [x] Store one complete JSON artifact per run and one generated aggregate JSON for each table.
- [x] Add manifest validation/tests so incomplete or inconsistent registered artifacts fail.
- [x] Preserve failed runs and exclusions in a run log; never silently drop a seed.

Suggested result ID:

```text
{model}__{strategy}__v{vocab_size}__{init}__e{epochs}__s{seed}
```

### R4. P0 - Vocabulary-size ablation

**Question:** Is 8K a justified operating point rather than an arbitrary choice?

- [x] Train the balanced mixed BPE tokenizer at **4K, 8K, 16K, and 32K**.
- [x] Hold corpora, balancing, normalization, special tokens, trainer settings, and test sets
  constant; vary only vocabulary size.
- [x] Evaluate every size on both conversational ASR and formal Twi test sets.
- [x] Report:
  - Token fertility and percentage reduction against each multilingual baseline.
  - Sequence-length distribution, not only the mean (median, p90, and p95).
  - Vocabulary utilization on held-out text.
  - Mean bytes/characters per token and special/unknown-token behavior.
  - Tokenizer model size and embedding-parameter cost at each target model hidden dimension.
- [x] Add bootstrap confidence intervals for fertility differences over test examples.
- [x] Produce a fertility-vs-vocabulary-size plot with separate ASR and formal series.
- [x] Produce a compact trade-off table covering fertility, embedding parameters/memory, and
  vocabulary utilization.
- [x] State the operating-point rule before inspecting results: prefer the smallest vocabulary
  on the performance plateau unless model quality shows a material disadvantage.
- [x] Use the result to retain 8K or change the model-integration vocabulary; do not justify 8K
  retrospectively.

**Result (July 30, 2026):** The predeclared operational form of the plateau rule selects the
smallest size within 1% relative of the best observed fertility in both domains. Only 32K
qualifies, so 32K is locked for the new extension and multi-seed experiments. This is explicitly a
boundary selection, not evidence that a plateau was observed before the upper bound; controlled
model-quality evidence may still override it. The historical 8K model ladder remains usable: the
fresh 8K tokenizer has the same vocabulary and identical encodings on both frozen tests as the
historical mixed tokenizer.

**Acceptance criteria**

- All four tokenizers are trained from the same frozen corpus/config.
- The result artifact and figure can be regenerated with one documented command.
- The paper explains how vocabulary size interacts with embedding dimension, model memory,
  sequence length, and the available fine-tuning data.

### R5. P0 - Vocabulary extension baseline

**Question:** Is full vocabulary replacement better than adding Akan-specific tokens while
preserving the base tokenizer and its multilingual knowledge?

**Implementation status (August 22, 2026):** The CPU-side extension contract, intrinsic
comparison, and controlled 0.6B three-seed GPU comparison are complete. The locked 32K mixed vocabulary contributes 26,156 novel tokens after
excluding eight shared special tokens and 5,836 exact Qwen collisions. All 151,669 original token
IDs remain stable after save/reload. The extension reduces fertility by 23.1% on ASR and 25.4% on
formal Twi versus the original Qwen tokenizer, while full replacement reduces it by 49.6% and
52.0%. Under the controlled model-quality protocol, extension is worse than full replacement on
BPB and chrF/chrF++ while requiring a much larger lexical checkpoint. The downstream-task portion
of the comparison remains the separate P1 gate in R9.

**Execution readiness (August 2, 2026):** The remaining controlled runs are frozen in the single
`config/revision_gpu_matrix.yaml` contract. The matrix runner expands stable IDs, executes exactly
one isolated run at a time, resumes from validated per-run JSON artifacts, and records its YAML
SHA-256 in every result. The aggregator refuses incomplete, stale, or configuration-mismatched
matrices. No GPU results are claimed until those artifacts exist.

- [x] Implement an **extension** strategy that adds Akan-specific tokens to the original model
  tokenizer without deleting its existing vocabulary.
- [x] Define extension budgets that make the comparison interpretable:
  - Primary comparison: add a fixed number of Akan tokens selected from the final mixed tokenizer.
  - Resource-matched comparison: report the resulting embedding parameter/memory increase.
  - If compute permits, include more than one extension budget to show the curve.
- [x] Exclude tokens already represented identically in the base vocabulary.
- [x] Specify token-selection rules, normalization compatibility, collision handling, special-token
  handling, embedding resizing, tied output-head behavior, and serialization/reload behavior.
- [x] Initialize new rows with mean-of-subword embeddings from the original tokenizer; optionally
  retain random initialization as a diagnostic arm.
- [x] Compare on one controlled anchor model first (`Qwen/Qwen3-0.6B`), using the same corpus,
  training steps, QLoRA recipe, seed set, and evaluation examples as full replacement.
- [ ] Compare **original tokenizer vs extension vs full replacement** on:
  - ASR and formal fertility.
  - Corrected full-coverage BPB.
  - chrF and chrF++ continuation quality.
  - Downstream-task performance.
  - Trainable/total parameters, embedding memory, throughput or processed tokens, and checkpoint
    size where available.
- [x] Add round-trip save/reload tests and tests confirming original token IDs remain stable.
- [x] Discuss the central trade-off: extension preserves multilingual vocabulary/knowledge but
  increases embeddings; replacement maximizes Akan efficiency but discards the original lexical
  interface.

**Frozen primary budget and resource trade-off:** Traverse the complete locked 32K candidate
vocabulary in learned-ID order, exclude specials and exact base collisions, and append all 26,156
remaining surfaces without normalization. Qwen's base tokenizer has 151,669 IDs and its model
preallocates 151,936 embedding rows. After hardware padding the extension uses 177,856 rows,
adding 25,920 actual model rows: 53.08M untied input/output parameters or 101.25 MiB in FP16. Its
total lexical interface is 694.75 MiB FP16 versus 125.00 MiB for 32K replacement. Thus extension
preserves the multilingual interface and cuts tokenization cost substantially, but replacement is
far more intrinsically efficient and much smaller. BPB, chrF/chrF++, downstream quality,
throughput, and checkpoint size must decide whether preservation is worth that cost.

**Acceptance criteria**

- The baseline changes only the tokenizer adaptation strategy.
- It runs through the same training/evaluation pipeline as replacement.
- The paper reports resource cost as well as quality, so the comparison is not based on BPB alone.

### R6. P0 - Multi-seed validation

**Question:** Are the initialization gains reliable or artifacts of a single run?

**Execution status (August 22, 2026):** All 15 frozen runs are complete and validate against
matrix SHA-256 `883c6c91347d22757616cd44e373ea8bb96abdc1182fb53f28305a184a117a5d`.
The per-run JSON artifacts are in `results/revision_v2/gpu_runs`, and
`results/revision_v2/gpu_matrix_aggregate.json` contains arm means, sample standard deviations,
per-seed deltas, and paired 95% t intervals. The
matrix contains 12 replacement runs (two model sizes × two initialization modes × three seeds)
plus three Qwen 0.6B extension/mean-subword runs. The Qwen 0.6B replacement/mean-subword arm is
shared between R5 and R6, avoiding duplicate jobs. Seeds are frozen at 17, 42, and 73.

- [x] Use **three training seeds** for the controlled Qwen scale endpoints:
  `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B`.
- [x] Run both `random` and `mean_subword` initialization for every model/seed combination.
- [x] Reuse the existing run only if its full configuration and seed can be reconstructed exactly;
  otherwise run three clean seeds.
- [x] Keep data order rules, training steps/epochs, effective batch size, learning rate, and
  evaluation examples fixed across arms.
- [x] Report each seed plus mean, standard deviation, and a paired confidence interval for:
  corrected full-coverage BPB, chrF, and chrF++.
- [x] Report the direction and magnitude of the paired mean-subword-minus-random effect per seed.
- [ ] Do not use seed-based significance language when there are too few independent runs; emphasize
  effect consistency and uncertainty.
- [x] If extension becomes a headline conclusion, run it with the same three seeds on the 0.6B
  anchor model.

**Minimum GPU matrix**

| Model | Strategy | Initialization | Seeds |
|---|---|---|---|
| Qwen3-0.6B | Full replacement | Random | 3 |
| Qwen3-0.6B | Full replacement | Mean-subword | 3 |
| Qwen3-1.7B | Full replacement | Random | 3 |
| Qwen3-1.7B | Full replacement | Mean-subword | 3 |
| Qwen3-0.6B | Vocabulary extension | Mean-subword | 3 if used for the main claim |

**Execution commands**

```bash
python scripts/run_revision_gpu_matrix.py validate
python scripts/run_revision_gpu_matrix.py status
python scripts/run_revision_gpu_matrix.py run --next
python scripts/aggregate_revision_gpu_matrix.py
```

Invoke `run --next` once per fresh GPU process until `status` reports all 15 runs complete. Each
run records the seed, corrected BPB, chrF/chrF++, Trainer throughput, processed non-padding tokens,
checkpoint bytes, and reload verification. Aggregation reports every seed, arm means/sample
standard deviations, and 95% paired t intervals for mean-subword-minus-random at both sizes and
extension-minus-replacement at 0.6B. The downstream task remains the separate P1 gate in R9.

### R7. P0 - Router: document it or demote it

**Decision (August 2, 2026):** Routing is demoted from a central contribution to a secondary
analysis. The frozen audit in `results/router_audit_revision_v2.json` reconstructs the exact
stratified split, estimator settings, per-class metrics, and confusion matrices. It also records
the decisive scope limitation: the task predicts source-corpus identity for separately collected
ASR and formal-text datasets, not ambiguous real-world domain membership. No mixed-domain or
code-switched challenge set exists, and latency/tokenizer-switching overhead has not been measured.

- [x] Document the existing router completely:
  - TF-IDF representation and parameters.
  - Classifier type and parameters.
  - Train/validation/test construction and class balance.
  - Heuristic baseline.
  - Accuracy, precision, recall, F1, and confusion matrix; explicitly mark routing overhead as
    unmeasured rather than implying a deployment result.
- [x] Explicitly acknowledge that the current corpora are nearly trivially separable and that this
  does not establish robust real-world routing.
- [x] Move the near-perfect in-domain result to a short secondary implementation note.
- [x] Remove language implying that current routing performance generalizes to ambiguous or
  code-switched inputs.
- [ ] If routing remains a contribution, build a challenge set containing short utterances,
  punctuation-poor text, code-switching, mixed registers, and out-of-domain Akan; document its
  construction and evaluate error impact on fertility/model behavior.
- [x] Otherwise, use the balanced mixed tokenizer as the primary deployment path and defer robust
  routing to future work.

**Audit result:** The historical 80/20 holdout contains 1,617 ASR and 9,000 formal-text examples;
the classifier makes one error (99.9906% accuracy; confusion matrix `[[1617, 0], [1, 8999]]`). On
the active external tests it makes two errors across 3,510 examples (99.9430%), while the heuristic
reaches 78.3191%. These values show that the implementation can identify these source corpora, but
they are not evidence of robust routing under ambiguous, mixed-register, or code-switched input.
The serialized scikit-learn 1.8.0 model also emits an inconsistent-version warning under the frozen
1.9.0 audit environment. The balanced mixed tokenizer is therefore the primary deployment path.

### R8. P1 - Clarify metrics and reproducibility details

#### BPB

- [ ] Define corrected full-coverage BPB at first use with the exact implemented formula.
- [ ] Explain token-level negative log-likelihood conversion to bits and division by the exact
  number of evaluated UTF-8 bytes.
- [ ] Explain BOS/context handling, chunk boundaries, labels/masking, final partial chunks, and how
  full coverage avoids the former truncation bias.
- [ ] State whether the implementation follows a published method; cite it if so, otherwise describe
  it as the project implementation rather than implying a standard correction.
- [ ] Add unit tests using hand-checkable logits/text and tests proving identical byte coverage
  across tokenizers.
- [ ] Keep raw perplexity only for comparisons sharing the same tokenizer.
- [ ] Add original-tokenizer base-model PPL/BPB context where the comparison is mathematically valid.

#### chrF/chrF++

- [ ] Document the fixed protocol already used: 512 held-out continuations, 48 prompt words,
  64 reference words, and 64 generated tokens.
- [ ] Record the exact decoding strategy and all generation parameters: greedy/sampling/beam,
  temperature, top-p, top-k, repetition penalty, stop tokens, and seed.
- [ ] Explain prompt construction, exclusions, normalization, corpus source, and why no training
  example overlaps the evaluation prompts/references.
- [ ] Add interpretable reference points: an unadapted base model, a simple copy/retrieval baseline
  if meaningful, and the strongest adapted model.
- [ ] Include several qualitative generations and explain that absolute chrF scores of 15-19 are
  modest even when arm-to-arm differences are consistent.

#### QLoRA/training

- [ ] Add a complete hyperparameter table for every model/arm:
  quantization bits/type, compute dtype, LoRA rank, alpha, dropout, target modules, bias mode,
  learning rate/scheduler, optimizer, weight decay, warmup, epochs/steps, batch size, gradient
  accumulation, maximum length, clipping, checkpointing, seed, and hardware.
- [ ] Export this table from result artifacts/configs so paper values cannot drift from executed runs.
- [ ] Report trainable parameters and the treatment of embeddings/output heads.

### R9. P1 - Downstream Akan task evaluation

**Goal:** Demonstrate that lower fertility and BPB translate into useful model behavior.

**Execution readiness (August 22, 2026):** The downstream protocol is implemented and frozen in
`config/downstream_afrisenti.yaml`. It pins the human-annotated Twi AfriSenti revision, audits the
official split duplication, evaluates both the 949-row official test surface and a 730-row clean
sensitivity surface, and expands to two unadapted bases plus nine adapted runs across seeds 17, 42,
and 73. The CPU-safe contract/tests and resumable Kaggle notebook are complete; the 11 GPU result
artifacts and aggregate table remain pending.

- [x] Select at least one licensed Akan task with a defensible train/dev/test split. Prefer two
  complementary tasks if data quality and compute permit (for example sentiment plus topic
  classification or named-entity recognition).
- [x] Before selection, verify language variety, label quality, license, dataset size, leakage risk,
  and whether a causal LM can be evaluated fairly.
- [x] Freeze prompt/template, label mapping, decoding/parsing, and primary metric before running.
- [ ] Evaluate at least:
  - The best-BPB adapted model.
  - The weakest-BPB adapted model or the controlled 0.6B anchor.
  - The corresponding unadapted base model where feasible.
  - Vocabulary extension and full replacement on the anchor model.
- [ ] Report task-appropriate metrics (for example macro-F1 plus accuracy), per-class results,
  invalid-output rate, confidence intervals, and error examples.
- [ ] Keep the conclusion narrow if the task set is small: evidence of functional utility, not
  broad Akan understanding.
- [ ] If no trustworthy labeled dataset is available, document the search and use a small,
  independently reviewed evaluation set with a clear annotation protocol; do not silently substitute
  an unvalidated synthetic benchmark.

### R10. P1 - Qualitative tokenizer analysis

- [ ] Select **5-10 representative Akan sentences** spanning:
  - Conversational/ASR noise and fillers.
  - Formal prose.
  - Agglutination or affixation.
  - Reduplication and compounding.
  - Diacritics/orthographic variants.
  - Named entities and code-switching.
- [ ] Show segmentation and token counts for XLM-R, mBERT, mT5, the final Akan BPE tokenizer, and
  the vocabulary-extension tokenizer.
- [ ] Annotate which linguistic/orthographic patterns cause fragmentation.
- [ ] Avoid claiming morphological correctness without review by an Akan linguist/speaker.
- [ ] Include a compact version in the paper and the full set in an appendix/artifact.

### R11. P2 - Additional tokenizer algorithm comparison

- [ ] Train SentencePiece Unigram and WordPiece tokenizers under comparable corpus, normalization,
  special-token, and vocabulary-size settings.
- [ ] Compare fertility, sequence-length distribution, vocabulary utilization, training time, and
  qualitative segmentations on both test regimes.
- [ ] Clearly distinguish algorithm effects from implementation/pre-tokenization effects.
- [ ] Treat this as fertility/segmentation evidence unless compute permits controlled model
  integration; do not imply downstream superiority from fertility alone.

### R12. P2 - Longer fine-tuning ablation

- [ ] On Qwen3-0.6B, compare 1 vs 3 epochs for random and mean-subword initialization using the same
  seed(s) and evaluation protocol.
- [ ] Track validation BPB during training to distinguish faster convergence from a lasting
  initialization advantage.
- [ ] Report overfitting indicators and compute cost.
- [ ] Use this result to qualify whether mean-subword helps mainly at initialization or remains
  beneficial after longer adaptation.

### R13. Paper revision checklist

#### Framing and claims

- [ ] Preserve the applied-systems framing but state it once, positively, rather than repeatedly
  defending the absence of a new segmentation algorithm.
- [ ] Condense repeated caveats and move technical qualifications to the limitations section or
  footnotes.
- [ ] Replace “eliminates the tokenization tax” with “substantially reduces” unless every defined
  baseline/domain supports the stronger wording.
- [ ] Separate four claim types:
  intrinsic fertility reduction, model likelihood/BPB, continuation quality, and task usefulness.
- [ ] Do not claim measured latency, memory, or deployment gains unless they are directly benchmarked.

#### Methods

- [ ] Justify the selected vocabulary operating point using the new ablation.
- [ ] Describe full replacement and extension precisely.
- [ ] Add exact BPB, chrF, QLoRA, seed, and router protocols.
- [ ] Explain dataset scale and the limits of generalization beyond the two training domains.

#### Results

- [ ] Add the vocabulary-size curve and trade-off table.
- [ ] Add extension-vs-replacement results.
- [ ] Replace single-run initialization claims with multi-seed aggregates and uncertainty.
- [ ] Add the downstream task table.
- [ ] Add qualitative tokenization examples.
- [ ] Define every abbreviation and sign convention in table captions, including `Red.` and
  `BPB gain = base BPB - adapted BPB`, where positive means improvement.
- [ ] Retain PPL only where tokenizers are identical and label it accordingly.

#### Related work

- [ ] Expand African-language tokenizer/model adaptation coverage, including the reviewer-suggested
  Adebara et al. (2023), Alabi et al. (2022), and Ogueji et al. (2021).
- [ ] Strengthen vocabulary-extension coverage, including FOCUS and relevant follow-up work.
- [ ] Add tokenizer-model co-adaptation and evaluation beyond fertility, including the
  reviewer-suggested Goldman et al. (2024) and Petrov et al. (2024).
- [ ] Add BPE-vs-Unigram motivation, including Bostrom and Durrett (2020).
- [ ] Discuss byte-level/token-free alternatives, including the reviewer-suggested Clark et al.
  (2022), Xue et al. (2022), and Yu et al. (2023).
- [ ] Verify every suggested citation exists, is relevant, and supports the accompanying claim
  before adding it.

#### Figures, tables, and writing

- [ ] Ensure Figs. 1-4 are embedded, legible, referenced in order, and visible in the review PDF.
- [ ] Add a vocabulary-size/fertility plot.
- [ ] Add a qualitative segmentation visualization/table.
- [ ] Add a BPB decomposition only if the proposed “sequence length vs probability calibration”
  components can be defined and measured rigorously; otherwise omit it.
- [ ] Split the long sentence identified in the introduction.
- [ ] Define corrected full-coverage BPB at first use.
- [ ] Standardize reference formatting and apply the venue’s author-list convention.
- [ ] Check all values and captions against generated aggregate JSON.

### R14. Ethics, licenses, and broader impact

- [ ] Add a 2-3 sentence broader-impact statement warning that poorly adapted Akan models can
  produce harmful or incorrect output in government, education, and healthcare settings.
- [ ] State that human evaluation and domain-specific safety validation are required before
  high-stakes deployment.
- [ ] Verify and record licenses/terms for WaxalNLP, Pristine-Twi, all downstream datasets, and
  every base model.
- [ ] Confirm that redistribution of derived tokenizers, checkpoints, and evaluation examples is
  permitted.
- [ ] Credit dataset creators and communities as requested by their licenses or documentation.
- [ ] Document privacy, consent, and sensitive-content considerations for any new challenge or
  downstream evaluation set.

### R15. Reproducibility and final QA

- [ ] Provide documented commands for every new experiment.
- [ ] Add CPU-safe smoke tests for new tokenizer and artifact paths.
- [ ] Add GPU notebook/config cells that consume the same checked-in experiment manifests.
- [ ] Regenerate the consolidated result artifact from raw per-run outputs.
- [ ] Run the full automated test suite and record the environment.
- [ ] Independently recalculate headline percentages, means, standard deviations, confidence
  intervals, and table rounding.
- [ ] Confirm all reported sample counts and model/vocabulary facts from source artifacts.
- [ ] Render the final paper and visually inspect every page for missing figures, clipping,
  overflow, unreadable labels, and broken references.
- [ ] Check that the abstract, conclusion, tables, and artifact README report the same headline
  numbers.
- [ ] Prepare a reviewer-response matrix mapping each concern to the manuscript section, code/config,
  artifact, and result that resolves it.

### R16. Concrete deliverables

- [ ] Frozen revision experiment manifest/configs.
- [ ] Four mixed BPE tokenizer artifacts: 4K, 8K, 16K, and 32K.
- [ ] Vocabulary-size aggregate JSON, table, and plot.
- [ ] Vocabulary-extension implementation, tests, tokenizer/model artifacts, and comparison table.
- [x] Multi-seed per-run JSON files and aggregate statistics for both Qwen scale endpoints.
- [ ] Fully documented or demoted router section, plus an optional challenge-set artifact.
- [ ] BPB definition/tests and generated QLoRA hyperparameter table.
- [ ] Downstream-task configuration, results, confidence intervals, and error analysis.
- [ ] Qualitative tokenization table with 5-10 examples.
- [ ] Optional Unigram/WordPiece comparison and 1-vs-3-epoch ablation.
- [ ] Revised manuscript, regenerated figures/tables, broader-impact statement, and checked licenses.
- [ ] Reviewer-response matrix and final reproducibility checklist.

---

## 0. Research Design & Road to Paper (Pre-review Baseline)

This section records the plan and completed evidence that existed before the July 24, 2026
review. The Reviewer Revision Plan above is now authoritative. The detailed Phase 2 material in
§16 remains as background; where any older section conflicts with the Reviewer Revision Plan,
the reviewer-driven plan wins.

### 0.1 Locked decisions

| Decision | Choice |
|---|---|
| **Venue / scope** | IEEE Ghana ICAST 2026; applied AI/NLP systems paper for an under-resourced Ghanaian language |
| **Downstream evidence** | Bits-per-byte (BPB) **plus** generation quality (chrF) |
| **ASR scope** | Keep the dual-regime (ASR + TTS) story; **fix the ASR test split** first |
| **Model evidence** | **5 runs across 4 model families + a scale step** (see §0.3 M3); current ICAST tables come from the executed split notebooks |
| **Edge deployment (2B)** | Optional for the paper; a light latency note if cheap, otherwise future work |

### 0.2 Thesis

> Specialized BPE tokenizers eliminate the tokenization tax for Akan, and this gain
> survives transfer into a real LLM — yielding a more efficient, deployable model —
> across model scales and families.

Phase 1 supports clause 1 (intrinsic fertility). The model-integration phase supports clause 2 (the gain holds
inside a model, measured fairly). A light edge/latency note grounds "efficient/deployable."

### 0.3 The road — 5 milestones

**M1 — Lock the design (done).** Thesis, claims, metrics, venue/scope above.

**M2 — Methodology hardening (do *before* more model runs).** In priority order:
1. **Bits-per-byte (BPB). ✅ Implemented.** Perplexity is **not** comparable across tokenizers
   with different vocabularies. `akan_bpe/model_integration.py` now computes BPB for **both** the
   base model (original tokenizer) and the fine-tuned model (Akan tokenizer) on the same eval
   bytes (`eval.bpb` in the result JSON), so the cross-tokenizer claim is honest. `--skip-base-bpb`
   opts out of the second model load. Fertility is kept as the intrinsic metric; eval_loss/
   perplexity remain as a within-tokenizer training signal. *Highest-leverage fix — landed before
   the model runs so they do not need redoing.*
2. **Embedding-init ablation. ✅ Implemented.** `--embedding-init-mode {random,mean_subword}` on
   the model-integration CLI; `mean_subword` initializes each Akan-vocab row from the mean of the
   base model's subword embeddings for that token's surface string (the modeling contribution, see
   §16.1 failure mode). Run `random` vs `mean_subword` as a clean A/B (one variable changed).
   Use `--device-mode smoke` for a tiny CPU-only pipeline check without running a full GPU run.
3. **Regenerate the ASR test split. ✅ Done.** The stale single-sample test split (three ASR files
   left over from different runs) was regenerated via `scripts/download.py` — which now **fails
   loudly on a truncated split** (`_assert_healthy_split`). The full WaxalNLP `aka_asr` stream is
   10,107 rows → a clean **8,085 / 1,011 / 1,011** 80/10/10 split. Because the ASR train set
   changed, the ASR + mixed tokenizers and the router were retrained and the Phase 1 fertility
   benchmark re-run on the fixed split. Updated headline: **~47% ASR / ~46% TTS** fertility
   reduction vs the best multilingual baseline (was ~52%/~47% on the unreliable single sample). The
   TTS corpus was untouched (pinned at 45,000/2,500/2,500).

**M3 — Model evidence (5 runs; complete for ICAST).** Chosen to support both the **scale** and the **family**
clauses of the thesis, and to span base-vocab size and pretraining multilinguality. Selection
criteria (defensible in the paper): (1) QLoRA-feasible on a free T4 for reproducibility;
(2) standard causal-LM pipeline so the swap+resize method is the controlled variable;
(3) spans scale within a family and ≥3 families across; (4) diverse base-vocab sizes;
(5) a multilinguality spread from English-centric to Africa-purpose-built; (6) license
transparency.

| Run | Model | Params | ~Base vocab | Axis / why it's in | License |
|---|---|---|---|---|---|
| **run-qwen-0.6b ✅** | `Qwen/Qwen3-0.6B` | 0.6B | ~151k | Scale anchor (low); proven path | Apache-2.0 |
| **run-qwen-1.7b ✅** | `Qwen/Qwen3-1.7B` | 1.7B | ~151k | Scale anchor (high) — **isolates scale**, family held constant | Apache-2.0 |
| **run-gemma-1b ✅** | `google/gemma-3-1b-pt` | ~1B | ~256k | Multilingual + largest base vocab → tax survives *even* a 256k vocab | Gemma (gated) |
| **run-llama-1b ✅** | `meta-llama/Llama-3.2-1B` | ~1.2B | ~128k | English-centric → biggest tax/gain; deployment-standard, seeds Phase 2B | Llama (gated) |
| **run-aya-base ✅** | `CohereLabs/tiny-aya-base` | 3.35B | TBD* | Africa-aware multilingual pretraining → does the gain hold *even here?* + GGUF edge tie-in | CC-BY-NC |

`*` Aya base-vocab size is not on the model card — **read it from the config before citing.**

The table below records the notebook-derived ladder results. The executed notebooks are the source
of truth, and `results/notebook-ladder-results.json` is the derived machine-readable artifact.

Corrected full-coverage BPB results, mean-of-subword arm:

| Run | Base fert. | Akan fert. | Token reduction | Base BPB | Akan BPB | Δ BPB |
|---|---:|---:|---:|---:|---:|---:|
| `run-qwen-0.6b-mixed` | 2.530 | 1.264 | 50.1% | 2.9523 | **1.2871** | +1.6652 |
| `run-qwen-1.7b-mixed` | 2.530 | 1.264 | 50.1% | 2.7556 | **1.2505** | +1.5051 |
| `run-gemma-1b-mixed` | 2.284 | 1.264 | 44.7% | 3.3908 | **1.2411** | +2.1496 |
| `run-llama-1b-mixed` | 3.073 | 1.264 | 58.9% | 2.4480 | **1.2368** | +1.2112 |
| `run-aya-base-mixed` | 2.975 | 1.264 | 57.5% | 2.7129 | **1.2432** | +1.4697 |

In the notebook-derived results, all five mean-of-subword runs beat their base model under
full-byte coverage. Full payloads, including random-init arms, generation samples, reload checks,
and chrF generation-quality results, are derived from `notebooks/run-full-light.ipynb` and
`notebooks/run-full-heavy.ipynb` into `results/notebook-ladder-results.json`.

What the set buys, in reviewer terms:
- **Scale axis:** Qwen3 0.6B → 1.7B (one variable changed).
- **Family axis:** 4 distinct families (Qwen, Gemma, Llama, Cohere/Aya) → kills the "Qwen quirk" objection.
- **Base-vocab spread:** 128k / 151k / 256k / TBD → the tax isn't an artifact of one vocab size.
- **Multilinguality spread:** weak (Llama) → moderate (Qwen) → strong (Gemma) → Africa-built (Aya).
  The headline line: *the weaker a model's native Akan support, the more our tokenizer helps —
  and it still helps even the strongest.*

**Why `tiny-aya-base`, not `-earth`:** the other four are base/pretrained models; `tiny-aya-earth`
is instruction-tuned + preference-aligned, which would confound a tokenizer-swap + embedding
retrain. `tiny-aya-base` shares the same Africa-aware multilingual pretraining without the SFT
confound, keeping the comparison apples-to-apples.

**Per-run reporting:** fertility reduction, **BPB** (base vs Akan tokenizer), and generation
samples. ≥2 seeds where T4 budget allows; single-seed is an acceptable, stated limitation at
ICAST tier if clearly stated as a limitation. The ICAST ladder uses the balanced mixed tokenizer
now that the ASR split is real and the methodology sanity check has passed.

**Run order — cheapest/safest first, so results bank before the risky run:**
run-qwen-1.7b (config clone) → run-gemma-1b → run-llama-1b → **run-aya-base last** (heaviest at 3.35B *and*
a custom Cohere architecture; its LoRA target-module names and the `colab-qlora` allowlist need
a model-specific check — budget engineering time, it is not a config-only clone). If T4 time or
the Aya integration bites, a complete 4-run paper still stands.

**M4 — Generation quality. ✅ Done.** **chrF** and **chrF++** (preferred over BLEU for
morphologically rich, low-resource languages) are reported on held-out Twi continuations in the
executed notebooks. The protocol uses 512 examples per arm, 48 prompt words, 64 reference words,
and 64 generated tokens. In the notebook-derived results, `mean_subword` improves chrF/chrF++ over
`random` in all five model runs.

**M5 — Write & submit to ICAST.** `report.md` → paper skeleton. Intro (tokenization tax) · Related work
(low-resource tokenization, African NLP) · Method (specialized BPE + tokenizer swap + embedding
init) · Results (fertility, BPB, chrF, routing) · Discussion/limitations · Future work (full
ladder + edge + cross-lingual). Figures from existing result JSON; code release largely ready.

### 0.4 Critical path

```
M2 (BPB + ASR split fix)  ← before any new model run, or they get redone
        ↓
M3 (5 runs: Qwen3-0.6B + Qwen3-1.7B + Gemma-3-1B + Llama-3.2-1B + tiny-aya-base, in BPB)
   └─ executed split notebooks consolidated in `results/notebook-ladder-results.json`
        ↓
M4 (chrF generation quality)
        ↓
M5 (write)
```

### 0.5 Explicitly deferred to future work (not in the paper)

- The stretch/reference tier: `microsoft/Phi-4-mini-instruct` (off-thesis, code/English-heavy,
  QLoRA-stretch on a T4) and `CohereLabs/aya-expanse-8b` (8B + non-commercial, reference-only)
- Phase 2B edge deployment as a full benchmark suite (Dell Latitude 7400)
- Cross-lingual transfer, additional Akan domains, labeled-task (QA/instruction) evaluation

### 0.6 Open items to pin before building

- **CFP deadline and formatting.** Confirm the ICAST deadline, page limit, tracks, template, and
  CMT requirements so M3/M4 are budgeted against a real date.
- **Generation eval protocol. ✅ Done.** Held-out Twi continuation scoring is fixed in the
  notebooks: 512 examples, 48 prompt words, 64 reference words, 64 max generated tokens, chrF/chrF++.
- **Aya config facts.** Read `tiny-aya-base`'s vocab size, hidden size, tied-embeddings flag, and
  LoRA target-module names from its config before it goes in any table or the `colab-qlora` path.
- **Gated-model access.** Gemma and Llama are gated on Hugging Face — accept their licenses and
  set up an HF token before the run-gemma-1b / run-llama-1b runs.

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
- Token fertility benchmarking (~47% ASR reduction, ~46% TTS reduction vs best baseline)
- Balanced mixed tokenizer (corpus upsampling — now genuinely differentiates domains)
- Heuristic router implementation
- ML source-corpus classifier (99.99% held-out accuracy; secondary analysis only)

**In progress / next phases (per the §0 paper plan):**
- Methodology hardening (M2) — bits-per-byte eval, embedding-init ablation, and ASR test split fix are complete.
- Model integration (M3) — **complete for the current ICAST evidence package.** The 5-run corrected full-coverage BPB ladder is preserved in the executed notebooks and consolidated in `results/notebook-ladder-results.json`.
- Generation quality (M4) — **complete.** chrF/chrF++ on held-out Twi continuations is in the same notebook-derived artifact.
- Edge deployment — optional for the paper (light latency note if cheap); full GGUF + Dell Latitude 7400 benchmarking is deferred to future work.

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

- Implemented heuristic-based router (77.6% on TTS test, 80.2% on ASR test)
- Trained ML source-corpus classifier (TF-IDF + Logistic Regression, 99.99% held-out accuracy
  on a stratified 80/20 split of 53,085 samples)
- Per-class F1: ASR 0.9997, TTS 0.9999
- On the formal-text source corpus, the ML route matches the always-TTS fertility strategy

**Status:** Implementation complete; demoted to secondary analysis because this source-corpus task
does not test ambiguous, mixed-register, or code-switched routing. See R7.

### 14.2 Incremental tokenizer variants

If basic A/B/C results are promising, the project can revisit staged corpus ideas such as:

- `TTS -> ASR -> TTS`
- `ASR -> TTS`

These should only be attempted after the simpler comparisons are complete.

### 14.3 Model integration (COMPLETE FOR M3)

> **Metric note:** the executed split notebooks are the source of truth for the current ladder.
> `results/notebook-ladder-results.json` is the derived artifact generated from their preserved
> JSON output blocks. The older `results/model-ladder-results.json` is historical.

The model-integration runs are complete. The repo contains:

- `akan_bpe/model_integration.py` — dataset prep, tokenizer/model loading, token-count comparison, LoRA/QLoRA setup, eval, generation samples, JSON artifact creation
- `scripts/model_integration.py` — CLI-driven experiment runner
- `notebooks/run-full-light.ipynb` and `notebooks/run-full-heavy.ipynb` — executed Kaggle/T4 split notebooks with full result blocks preserved
- `scripts/extract_notebook_results.py` — extractor for `results/notebook-ladder-results.json`
- `tests/test_model_integration.py` — CPU-safe orchestration and artifact-contract coverage

**Current interpretation:** M3 and M4 are complete in the notebook-derived artifact. The 5-run
ladder spans Qwen, Gemma, Llama, and Aya; all five mean-of-subword arms beat random on BPB and
chrF/chrF++, and all five beat their base models on corrected full-coverage BPB. The generation
quality protocol is fixed at 512 held-out continuation examples per arm, 48 prompt words, 64
reference words, and 64 max generated tokens.

**Default tokenizer:** `scripts/model_integration.py` now defaults to the balanced
`models/mixed_tokenizer.json`, replacing the earlier TTS-only integration default. New reruns
write `-mixed` result IDs (e.g. `run-qwen-0.6b-mixed`, `run-qwen-0.6b-mixed-meansub`) so
previous TTS-tokenizer artifacts remain available for comparison.

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
6. ✅ ML source-corpus classifier (99.99% held-out accuracy; secondary analysis per R7)
7. ✅ End-to-end notebook (notebooks/train_eval.ipynb)

**Conclusion:** Specialization is real — the v2 checkpoint preserves the ~47% ASR and ~46%
formal-Twi headline reductions after removing the documented leaked ASR test row. The active ASR
evaluation contains 1,010 samples. The balanced 8K mixed tokenizer remains a viable historical
deployment choice, while the tokenizer-only ablation locks 32K for the next new reviewer
experiments under the predeclared rule.

---

## 16. Phase 2: Next Steps

Phase 1 answered the tokenizer question. Phase 2 asks whether those gains translate to a real model.

### 16.1 Model Integration

**Goal:** Verify that fertility reduction translates into measurable downstream benefit — faster inference, lower perplexity, or better generation — not just a smaller token count.

**Current status:** all five runs are complete — QLoRA runs executed end-to-end on Kaggle/T4 and
preserved in the executed split notebooks. Full payloads are consolidated in
`results/notebook-ladder-results.json`, derived from notebook output blocks. The repo contains:

- `akan_bpe/model_integration.py` for dataset prep, tokenizer/model loading, token-count comparison, LoRA/QLoRA setup, eval, generation samples, and JSON artifact creation
- `scripts/model_integration.py` for one CLI-driven experiment run
- `notebooks/run-full-light.ipynb` and `notebooks/run-full-heavy.ipynb` — executed split notebooks with full JSON blocks preserved in-notebook
- `scripts/extract_notebook_results.py` for rebuilding the consolidated notebook-derived artifact
- `tests/test_model_integration.py` for CPU-safe orchestration and artifact-contract coverage

**Hardware baseline:** Free Kaggle/Colab GPU, typically T4/P100-class. Train and evaluate the smaller models first; treat larger models as QLoRA-only or reference-only unless paid GPU access is available.

**Model ladder (background — see §0.3 M3 for the authoritative paper set):** The paper runs a
**5-model set** — Qwen3-0.6B, Qwen3-1.7B, Gemma-3-1B, Llama-3.2-1B, and tiny-aya-base — chosen
to span scale, family, base-vocab size, and multilinguality. The original six-rung ladder below
remains as a longer-term map; only the **stretch/reference tier** (Phi-4-mini,
aya-expanse-8b) is deferred to future work.

> Current per-run metrics are not duplicated here; use `results/notebook-ladder-results.json`,
> regenerated from the executed notebooks, for the authoritative table values.

Completed paper set: `run-qwen-0.6b`, `run-qwen-1.7b`, `run-gemma-1b`, `run-llama-1b`, and
`run-aya-base`. Deferred stretch/reference tier: `microsoft/Phi-4-mini-instruct` or
`CohereLabs/aya-expanse-8b`.

`Qwen2.5-0.5B` remains a fallback only if Qwen3 tooling causes friction. Tiny Aya Earth replaces Aya Expanse 8B as the primary Aya-family candidate because it is smaller, Africa/West Asia-focused, and designed for local deployment under realistic compute constraints.

**Steps:**

1. **Choose tokenizer to integrate** — use the balanced mixed tokenizer for the paper ladder. It preserves the single-tokenizer deployment story while reflecting the fixed ASR split and mixed-tokenizer sanity rerun.

2. **Resize token embeddings**
   ```python
   from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

   base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
   new_tokenizer = PreTrainedTokenizerFast(tokenizer_file="models/mixed_tokenizer.json")
   base_model.resize_token_embeddings(len(new_tokenizer))
   ```
   New token embeddings initialize randomly; existing tokens that map to the new vocab keep their weights where possible.

3. **Re-tokenize training data** — run `pristine_twi_train.jsonl` through the new tokenizer to produce training inputs for fine-tuning.

4. **Fine-tune** — LoRA is the practical choice on limited hardware. Full fine-tune only if GPU VRAM allows.
   - Library: `peft` + `transformers` Trainer or `trl` SFTTrainer
   - Target modules: attention Q/K/V projections
   - Rank: r=8 or r=16 to start

5. **Evaluate**
   - **Bits-per-byte (BPB)** on `pristine_twi_test.jsonl` — compare base model (original
     tokenizer) vs fine-tuned (new tokenizer). **Use BPB, not raw perplexity:** perplexity is
     not comparable across tokenizers with different vocabularies, so the cross-tokenizer
     claim must rest on a tokenizer-agnostic metric (see §0.3 M2.1).
   - **Generation quality** — **chrF** (preferred over BLEU for morphologically rich,
     low-resource Akan) on held-out Twi continuations; small qualitative rubric as backup.
   - **Inference speed** — tokens/second before and after to quantify the fertility gain in practice

6. **Record experiment output** — save one structured JSON per run under `results/`, including model ID, tokenizer path, dataset paths, fertility, perplexity, generation samples, timing, hardware, and memory notes.

**First real run path:** Use the split notebooks (`notebooks/run-full-light.ipynb` and
`notebooks/run-full-heavy.ipynb`) to install `.[dev,train]` plus `bitsandbytes`, verify
`data/pristine_twi_train.jsonl`, `data/pristine_twi_test.jsonl`, and `models/mixed_tokenizer.json`,
then call `scripts/model_integration.py` in `colab-qlora` mode.

**Success criterion:** Fine-tuned model with new tokenizer matches or exceeds base model perplexity on Akan test text, with fewer tokens processed per sample.

**Failure mode to watch for:** If BPB is significantly worse after embedding resize, the
initialization strategy needs work (e.g., averaging subword embeddings from the original vocab
that cover similar character sequences). For the paper this is promoted to a deliberate
**embedding-init ablation** (random vs mean-of-subword) — see §0.3 M2.2 — and is the modeling
contribution rather than just a risk to monitor.

#### 16.1.1 run-qwen-1.7b — ✅ done (`Qwen/Qwen3-1.7B`)

Completed on Kaggle/T4 in the heavy split notebook. The run-qwen-0.6b machinery cloned cleanly:
allowlist extended to `Qwen/Qwen3-1.7B`, TTS tokenizer + dataset paths kept, and
`CUDA_VISIBLE_DEVICES` pinned per arm to avoid the DataParallel + QLoRA crash on T4×2. Current
metrics are in `results/notebook-ladder-results.json`.

#### 16.1.2 run-gemma-1b — ✅ done (`google/gemma-3-1b-pt`)

Completed on Kaggle/T4 in the heavy split notebook. Different family, largest base vocab
(~256k), PT checkpoint to avoid the SFT confound. Current metrics are in
`results/notebook-ladder-results.json`; this rung addresses the "Qwen quirk" and
"small-vocab artifact" objections.

#### 16.1.3 run-llama-1b — ✅ done (`meta-llama/Llama-3.2-1B`)

Completed on Kaggle/T4 in the light split notebook. This is the English-centric rung and remains
the deployment-standard model for later edge work. Current metrics are in
`results/notebook-ladder-results.json`.

#### 16.1.4 run-aya-base — ✅ done (`CohereLabs/tiny-aya-base`) — final rung, M3 complete

Completed on Kaggle/T4 in the light split notebook. This is the Africa-aware rung (3.35B, custom
Cohere arch, run last). Current metrics are in `results/notebook-ladder-results.json`. This closes
the 5-run M3 set.

#### 16.1.5 Truncation-corrected BPB — fixed, recovered, and tracked

The original BPB metric truncated each text to `max_length-1` tokens but divided by the full byte
count, deflating BPB for high-fertility base tokenizers most — the cause of the negative
run-llama-1b / run-aya-base signs. Status:

1. **Diagnosed.** Truncation diagnostics (now retired) showed the high-fertility bases scored only
   ~30–40% of each text's bytes while being credited with 100%, exactly deflating their BPB.
2. **Fixed in the library.** `compute_bpb_metrics` (`akan_bpe/model_integration.py`) now scores both
   base and experiment with **full byte coverage** via `compute_model_bpb_full` (non-overlapping
   chunks); `compute_model_bpb_sliding` is available as a cross-check. Every model is scored on 100%
   of identical content, so high-fertility bases are no longer flattered.
3. **Recovered and tracked.** The split notebooks preserved full JSON payloads in stdout blocks.
   Those payloads are consolidated directly into `results/notebook-ladder-results.json` by
   `scripts/extract_notebook_results.py`.
4. **Unaffected by the truncation fix.** The fertility reductions (45–59%) and the embedding-init
   ablation pattern (mean_subword wins every rung in the notebook-derived artifact) do not depend
   on the old truncated BPB calculation.

**ASR test split — M2 fix, ✅ done.** The full WaxalNLP `aka_asr` stream originally produced an
8,085 / 1,011 / 1,011 split. The revision audit found one normalized sentence shared by train row
4,250 and historical test row 368 under different IDs. Revision v2 preserves train and validation,
removes only that test row, and freezes the active split at **8,085 / 1,011 / 1,010**. Existing
tokenizers were not retrained because their training inputs did not change; only affected
fertility and router evaluations were rerun. The v1-v2 checkpoint found only the expected
one-example numerical effect and retained every headline interpretation and the historical model
ladder.

---

### 16.2 Edge Deployment

> **Paper scope note (see §0.1):** Full edge benchmarking is **optional for the ICAST
> paper** and otherwise deferred to future work. For the submission, fold in at most a light
> latency / tokens-per-second note if it is cheap to obtain. The full suite below is the
> longer-term plan.

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
   - Fine-tuned model, balanced mixed Akan tokenizer
   - Fine-tuned model + ML router (dynamic tokenizer selection)

**Success criterion:** Fine-tuned Akan model generates tokens faster (more tokens/second or fewer tokens per prompt) than the base model on Akan input, with acceptable RAM footprint for the target hardware.

---

### 16.3 Sequencing

Paper critical path (authoritative — see §0.3/§0.4):

```
Phase 1 (DONE)
    └── M2 Methodology hardening
    │       ├── BPB metric (before any new model run)
    │       ├── Embedding-init ablation (random vs mean-of-subword)
    │       └── Fix ASR test split + re-run Phase 1 fertility benchmark
    └── M3 Model evidence (5 runs, reported in BPB; complete in executed notebooks)
    │       ├── run-qwen-0.6b
    │       ├── run-qwen-1.7b (scale step)
    │       ├── run-gemma-1b (multilingual, 256k vocab)
    │       ├── run-llama-1b (English-centric)
    │       └── run-aya-base (Africa-aware, 3.35B)
    │           └── consolidated in results/notebook-ladder-results.json
    └── M4 Generation quality (chrF on held-out Twi; complete)
    └── M5 Write & submit (IEEE Ghana ICAST 2026)

Deferred to future work:
    ├── stretch/reference tier (Phi-4-mini, aya-expanse-8b)
    └── Phase 2B edge deployment (full GGUF + Dell Latitude 7400 suite)
```

Longer-term map (background): the full model-integration ladder runs all five rungs plus the
stretch tier, followed by Phase 2B (GGUF export, bundle router, benchmark on Dell Latitude 7400). Phase 2B is blocked on a working
fine-tuned model artifact; for the paper it is optional/future per §16.2.
