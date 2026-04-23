# RAG Evaluation Matrix Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a dependency-free RAG evaluation matrix executor that produces comparable reports for A0/A1/B1/G0/G1/G2/G3.

**Architecture:** Add one focused script under `scripts/` with pure metric helpers, variant orchestration, retrieval invocation, and report writing. Add one focused unit test module for metric and comparison behavior. Keep normal RAG runtime behavior unchanged.

**Tech Stack:** Python 3.12, existing backend modules, `unittest`, JSONL report files, existing `.venv`.

---

### Task 1: Inspect Retrieval Entry Points

**Files:**
- Read: `backend/rag_utils.py`
- Read: `backend/rag_pipeline.py`
- Read: `tests/test_rag_utils.py`

- [ ] **Step 1: Locate the retrieval function used by existing tests**

Run:

```powershell
Select-String -Path backend\rag_utils.py,backend\rag_pipeline.py,tests\test_rag_utils.py -Pattern "retrieve|hybrid|trace|fallback_required" -Context 2,4
```

Expected: identify the callable that accepts a query and returns retrieved chunks plus metadata/trace.

- [ ] **Step 2: Record required environment toggles**

The executor will set these before loading backend modules in subprocess mode:

```text
EVAL_RETRIEVAL_TEXT_MODE
STRUCTURE_RERANK_ENABLED
CONFIDENCE_GATE_ENABLED
LOW_CONF_TOP_MARGIN
LOW_CONF_ROOT_SHARE
LOW_CONF_TOP_SCORE
ENABLE_ANCHOR_GATE
```

Expected: no code changes in this step.

### Task 2: Add Metric Helper Tests

**Files:**
- Create: `tests/test_evaluate_rag_matrix.py`
- Target: pure helper functions from `scripts/evaluate_rag_matrix.py`

- [ ] **Step 1: Write tests for matching and metrics**

Create tests that import these planned helpers:

```python
from scripts.evaluate_rag_matrix import (
    compare_sample_rank,
    compute_retrieval_metrics,
    first_relevant_rank,
)
```

Test cases:

```python
def test_compute_metrics_uses_root_anchor_keyword_matches():
    docs = [
        {"chunk_id": "leaf-a", "root_chunk_id": "wrong", "text": "noise"},
        {"chunk_id": "leaf-b", "root_chunk_id": "root-1", "section_title": "第二条", "text": "自然人 民事关系"},
    ]
    metrics = compute_retrieval_metrics(
        docs,
        expected_chunk_ids=[],
        expected_root_ids=["root-1"],
        expected_anchors=["第二条"],
        expected_keywords=["自然人"],
        top_k=5,
    )
    assert metrics["hit_at_5"] is True
    assert metrics["root_hit_at_5"] is True
    assert metrics["anchor_hit_at_5"] is True
    assert metrics["keyword_hit_at_5"] is True
    assert metrics["first_relevant_rank"] == 2
    assert metrics["mrr"] == 0.5
    assert metrics["context_precision_id_at_5"] == 0.5
    assert metrics["irrelevant_context_ratio_at_5"] == 0.5
```

```python
def test_compare_sample_rank_counts_win_loss_tie():
    old = {"hit_at_5": False, "first_relevant_rank": None}
    new = {"hit_at_5": True, "first_relevant_rank": 3}
    assert compare_sample_rank(old, new) == "win"
    assert compare_sample_rank(new, old) == "loss"
    assert compare_sample_rank(new, {"hit_at_5": True, "first_relevant_rank": 3}) == "tie"
```

- [ ] **Step 2: Run tests and verify they fail before implementation**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_evaluate_rag_matrix -v
```

Expected: import failure because `scripts/evaluate_rag_matrix.py` does not exist yet.

### Task 3: Implement Pure Evaluation Helpers

**Files:**
- Create: `scripts/evaluate_rag_matrix.py`

- [ ] **Step 1: Add helper functions**

Implement:

```python
def first_relevant_rank(docs, expected_chunk_ids=None, expected_root_ids=None, expected_anchors=None, expected_keywords=None, top_k=5):
    ...

def compute_retrieval_metrics(docs, expected_chunk_ids=None, expected_root_ids=None, expected_anchors=None, expected_keywords=None, top_k=5):
    ...

def compare_sample_rank(old_metrics, new_metrics):
    ...
```

The helpers must not import live backend services.

- [ ] **Step 2: Run metric tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_evaluate_rag_matrix -v
```

Expected: tests pass.

### Task 4: Implement Dataset and Reporting Helpers

**Files:**
- Modify: `scripts/evaluate_rag_matrix.py`

- [ ] **Step 1: Add JSONL loading**

Implement:

```python
def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    ...
```

It skips blank lines, preserves order, and supports `--limit`.

- [ ] **Step 2: Add report writers**

Implement:

```python
def write_jsonl(path: Path, rows: list[dict]) -> None:
    ...

def summarize_results(rows: list[dict], variants: list[str]) -> dict:
    ...

def render_summary_markdown(summary: dict) -> str:
    ...
```

Expected outputs include per-variant aggregates, diagnostic category counts, and paired comparisons.

### Task 5: Implement Retrieval Invocation

**Files:**
- Modify: `scripts/evaluate_rag_matrix.py`

- [ ] **Step 1: Add backend path setup**

At script startup, insert `backend/` into `sys.path`, then import runtime retrieval only inside functions that execute evaluation.

- [ ] **Step 2: Add one-sample evaluation**

Implement:

```python
def evaluate_sample(record: dict, variant: str, top_k: int) -> dict:
    ...
```

The result row includes sample id, query, expected fields, retrieved chunk summaries, trace, metrics, diagnostic result, latency, and error.

- [ ] **Step 3: Continue on sample-level errors**

If one query fails, write a row with:

```json
{
  "error": "...",
  "metrics": {"error_rate": 1.0}
}
```

Expected: the script keeps evaluating later samples.

### Task 6: Implement Matrix Orchestration

**Files:**
- Modify: `scripts/evaluate_rag_matrix.py`

- [ ] **Step 1: Add CLI**

Support:

```powershell
.venv\Scripts\python.exe scripts\evaluate_rag_matrix.py --dataset .jbeval\datasets\rag_tuning_derived.jsonl --limit 3 --variants B1 --run-id smoke
```

Options:

```text
--dataset
--output-root
--run-id
--limit
--top-k
--variants
--skip-reindex
--allow-destructive-reindex
```

- [ ] **Step 2: Add variant config**

Represent A0/A1/B1/G0/G1/G2/G3 as dictionaries containing env vars and reindex requirement.

- [ ] **Step 3: Add reindex execution**

For A0 and A1, call:

```powershell
.venv\Scripts\python.exe scripts\reindex_knowledge_base.py
```

Stop the matrix if reindex fails.

- [ ] **Step 4: Write report files**

Write:

```text
.jbeval/reports/<run_id>/results.jsonl
.jbeval/reports/<run_id>/summary.json
.jbeval/reports/<run_id>/summary.md
.jbeval/reports/<run_id>/config.json
```

### Task 7: Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_evaluate_rag_matrix tests.test_rag_diagnostics -v
```

Expected: all tests pass.

- [ ] **Step 2: Run full unit suite**

Run:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Run smoke evaluation**

Run:

```powershell
.venv\Scripts\python.exe scripts\evaluate_rag_matrix.py --dataset .jbeval\datasets\rag_tuning_derived.jsonl --variants B1 --limit 2 --skip-reindex --run-id rag-matrix-smoke
```

Expected: report directory exists with all four files.

- [ ] **Step 4: Run approved matrix**

Run:

```powershell
.venv\Scripts\python.exe scripts\evaluate_rag_matrix.py --dataset .jbeval\datasets\rag_tuning_derived.jsonl --variants A0,A1,B1,G0,G1,G2,G3 --allow-destructive-reindex --run-id rag-matrix-full-YYYYMMDD-HHMM
```

Expected: final report contains all variants, paired comparisons, and diagnostic counts.

### Task 8: Code Self-Review

**Files:**
- Review: `scripts/evaluate_rag_matrix.py`
- Review: `tests/test_evaluate_rag_matrix.py`

- [ ] **Step 1: Run git diff review**

Run:

```powershell
git diff -- scripts\evaluate_rag_matrix.py tests\test_evaluate_rag_matrix.py
```

Expected: no unrelated changes.

- [ ] **Step 2: Use code-review skill**

Review for:

- sample-level error isolation
- deterministic metric correctness
- no secrets printed
- no dependency additions
- no RAG runtime behavior changes

Expected: fix any blocking issue before final report.
