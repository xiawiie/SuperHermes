# backend.rag.utils and Pipeline Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the stateful backend.rag.utils and LangGraph pipeline implementations into the `backend/rag` package without breaking legacy imports or monkeypatch behavior.

**Architecture:** `backend.rag.utils.py` owns the old `rag_utils.py` implementation. `backend/rag/pipeline.py` owns the old `rag_pipeline.py` implementation. Root files remain `sys.modules` compatibility aliases.

**Tech Stack:** Python 3.12, LangGraph, pytest, ruff, compileall.

---

## Scope

Moved:

- `backend/rag_utils.py` -> `backend.rag.utils.py`
- `backend/rag_pipeline.py` -> `backend/rag/pipeline.py`

Kept aliases:

- `backend/rag_utils.py`
- `backend/rag_pipeline.py`

Updated imports:

- `backend/chat/agent.py`
- `backend/chat/tools.py`
- `backend/evaluation/answer_eval.py`
- `scripts/evaluate_rag_matrix.py`
- `frontend/ui-redesign.test.mjs`

## Execution Evidence

Pre-change behavior lock:

```text
uv run pytest tests/test_rag_utils.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py tests/test_fallback_disabled_routing.py -q
34 passed, 1 warning
```

Expected failing boundary test before implementation:

```text
uv run pytest tests/test_backend.rag.utils_pipeline_boundaries.py -q
FAILED with ModuleNotFoundError for backend.rag.utils/rag.pipeline
```

Targeted post-move gate:

```text
uv run pytest tests/test_backend.rag.utils_pipeline_boundaries.py tests/test_rag_utils.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py tests/test_fallback_disabled_routing.py -q
37 passed, 1 warning
```
