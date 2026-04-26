# Backend Shared and Document Package Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move pure shared helpers and document parsing implementation out of the backend root while preserving existing legacy imports.

**Architecture:** `backend/shared/*` owns generic helpers with no business dependency. `backend/documents/*` owns document parsing and chunk preparation. Root modules stay as `sys.modules` compatibility aliases so existing tests, scripts, and runtime imports keep working.

**Tech Stack:** Python 3.12, pytest, ruff, compileall.

---

## Scope

Move implementations:

- `backend/json_utils.py` -> `backend/shared/json_utils.py`
- `backend/filename_normalization.py` -> `backend/shared/filename_normalization.py`
- `backend/document_loader.py` -> `backend/documents/loader.py`

Keep compatibility aliases:

- `backend/json_utils.py`
- `backend/filename_normalization.py`
- `backend/document_loader.py`

Update internal imports:

- `backend/answer_eval.py`
- `backend/query_plan.py`
- `backend/rag_utils.py`
- `backend/services/document_service.py`

## Cleanup Plan

Smells addressed:

1. **Root directory noise:** pure helpers and document parsing live beside app entrypoints.
2. **Boundary ambiguity:** filename and JSON helpers are shared utilities, not RAG or HTTP code.
3. **Document ownership:** `DocumentLoader` belongs under a document package, not the backend root.

Order:

1. Run targeted tests for the current behavior.
2. Add package-boundary tests that fail before implementation.
3. Move files into `shared` and `documents`.
4. Replace root files with `sys.modules` aliases.
5. Update internal imports to package paths.
6. Run targeted tests, then full gates.

## Verification Gates

Targeted behavior lock:

```powershell
uv run pytest tests/test_answer_eval.py tests/test_filename_normalization.py tests/test_document_loader.py tests/test_document_service.py tests/test_query_plan_parser.py tests/test_rag_utils.py -q
```

Post-move targeted gate:

```powershell
uv run pytest tests/test_backend_shared_documents_boundaries.py tests/test_answer_eval.py tests/test_filename_normalization.py tests/test_document_loader.py tests/test_document_service.py tests/test_query_plan_parser.py tests/test_rag_utils.py -q
```

Final gates:

```powershell
uv run pytest tests/ -q
uv run ruff check backend/ scripts/ tests/
uv run python -m compileall backend scripts
node --check frontend\script.js
node --check frontend\src\api.js
node --check frontend\src\messages.js
node frontend\ui-redesign.test.mjs
```

## Execution Evidence

Pre-change behavior lock:

```text
uv run pytest tests/test_answer_eval.py tests/test_filename_normalization.py tests/test_document_loader.py tests/test_document_service.py tests/test_query_plan_parser.py tests/test_rag_utils.py -q
59 passed, 1 warning
```

Expected failing boundary test before implementation:

```text
uv run pytest tests/test_backend_shared_documents_boundaries.py -q
FAILED with ModuleNotFoundError for shared/documents packages
```

Targeted post-move gate:

```text
uv run pytest tests/test_backend_shared_documents_boundaries.py tests/test_answer_eval.py tests/test_filename_normalization.py tests/test_document_loader.py tests/test_document_service.py tests/test_query_plan_parser.py tests/test_rag_utils.py -q
61 passed, 1 warning
```
