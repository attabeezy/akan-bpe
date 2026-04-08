# Plan: Align Project Documentation with Implementation

This plan addresses inconsistencies between the project's documentation (`README.md`, `plan.md`, `code_spec.md`) and its actual implementation (`somax/`, `scripts/`).

## Objective
Standardize project naming, architectural descriptions, and script references to ensure a cohesive and accurate developer experience.

## Key Changes

### 1. Documentation Cleanup
*   **Standardize Naming**: Use `SOMAX` as the project name and `somax` as the package name consistently.
*   **Architecture Alignment**: Update `README.md` and `code_spec.md` to reflect the "Unified Vocabulary" approach (single tokenizer instance) used in `tokenizer.py`, rather than implying multiple tokenizer models.
*   **Script References**: Update all docs to use actual filenames:
    *   `train_bpe.py`, `train_lora.py`, `export_gguf.py` (removing numbered prefixes in `code_spec.md`).
    *   `benchmark_fertility.py` and `benchmark_inference.py` (replacing `benchmark.py` and `benchmark_edge.py`).
*   **Router Heuristics**: Update `code_spec.md` to match the full list of markers in `router.py`.
*   **Quick Start Paths**: Correct example commands in `README.md` to use accurate default output paths (e.g., `models/tokenizers/akan/`).

### 2. Code Consistency (Minor)
*   Ensure `DualCoreTokenizer` and `WAXALRouter` docstrings are fully aligned with the updated architectural description.

## Implementation Steps

### Phase 1: Update `code_spec.md`
*   Modify directory structure tree.
*   Update `WAXALRouter` class snippet with correct markers.
*   Update `DualCoreTokenizer` class snippet to show single `_tokenizer` usage.
*   Rename script references.

### Phase 2: Update `README.md`
*   Fix "Quick Start" commands.
*   Clarify "Dual-Stream" means dual-weight-sets with a shared vocabulary.
*   Fix benchmark command examples.

### Phase 3: Update `plan.md`
*   Ensure the high-level vision matches the implementation.

## Verification & Testing
*   Verify all script names mentioned in docs exist in the `scripts/` directory.
*   Verify all paths in "Quick Start" are reachable or correctly represent default output locations.
*   Ensure `pyproject.toml` package name matches the directory `somax`.
