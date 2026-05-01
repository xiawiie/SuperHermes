# RAG Profile Renaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move RAG evaluation and runtime naming from historical experiment names to the short `K/I/M/A/dtype` profile scheme without breaking existing behavior.

**Architecture:** Add a pure naming resolver, switch evaluation entrypoints and report display to new profile names, then wire runtime dtype/device parsing through the same resolver. Historical names remain readable only as migration aliases until final retirement.

**Tech Stack:** Python 3.12, pytest/unittest, FastAPI backend modules, existing RAG evaluation scripts.

---

### Task 1: Naming Resolver

**Files:**
- Create: `backend/rag/profile_naming.py`
- Test: `tests/test_rag_profile_naming.py`

- [ ] Add a pure resolver for `K1/K2/K3`, `I2`, `M0`, `A0/A1/A2`, and `fp16/bf16/fp32`.
- [ ] Verify `K2` resolves to historical alias `V3Q` and `K3` resolves to `V3Q_OPT`.
- [ ] Verify dtype mapping rejects unknown values.

### Task 2: Evaluation Variants

**Files:**
- Modify: `scripts/rag_eval/variants.py`
- Modify: `scripts/evaluate_rag_matrix.py`
- Test: `tests/test_evaluate_rag_matrix.py`

- [ ] Make `K1/K2/K3` the new executable variant names.
- [ ] Keep historical names as aliases that resolve to the new names.
- [ ] Add profile metadata and config hashes to report fingerprints.
- [ ] Verify `K2` and `V3Q` resolve to equivalent config.

### Task 3: Report Readability

**Files:**
- Modify: `scripts/rag_eval/reporting.py`
- Test: `tests/test_evaluate_rag_matrix.py`

- [ ] Render profile names as primary labels in summaries.
- [ ] Keep historical aliases in build/config metadata, not as primary names.
- [ ] Verify summary markdown displays `K2` and preserves historical alias context.

### Task 4: Runtime Dtype And Device

**Files:**
- Modify: `backend/rag/utils.py`
- Test: `tests/test_rag_utils.py`

- [ ] Resolve `RAG_DTYPE=fp16/bf16/fp32` to the torch dtype strings used by rerank.
- [ ] Keep explicit `RERANK_TORCH_DTYPE` higher priority than `RAG_DTYPE`.
- [ ] Verify rerank metadata reports the resolved dtype.

### Task 5: Documentation And Verification

**Files:**
- Modify: `docs/rag-profile-naming.md`
- Modify as needed: `README.md`, `eval/docs/rag-evaluation-runbook.md`, `eval/docs/rag-v3.1-report.md`

- [ ] Keep historical reports readable without rewriting original historical facts.
- [ ] Update new recommended commands to use `K2/K3`.
- [ ] Run targeted tests, smoke retrieval, smoke graph, and full gold RAG performance evaluations.
