# Backend Runtime Helper Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move query planning, answer evaluation, and conversation storage implementations out of the backend root while preserving legacy imports.

**Architecture:** `backend/rag/query_plan.py` owns query-scope planning for retrieval. `backend/evaluation/answer_eval.py` owns answer-generation/evaluation helpers used by offline evaluation. `backend/infra/db/conversation_storage.py` owns chat-history persistence. Root modules remain compatibility aliases.

**Tech Stack:** Python 3.12, LangChain, SQLAlchemy, pytest, ruff, compileall.

---

## Scope

Move implementations:

- `backend/query_plan.py` -> `backend/rag/query_plan.py`
- `backend/answer_eval.py` -> `backend/evaluation/answer_eval.py`
- `backend/conversation_storage.py` -> `backend/infra/db/conversation_storage.py`

Keep compatibility aliases:

- `backend/query_plan.py`
- `backend/answer_eval.py`
- `backend/conversation_storage.py`

Update internal imports:

- `backend/agent.py`
- `backend/rag_utils.py`
- `scripts/evaluate_rag_matrix.py`

Do not move:

- `agent.py`
- `tools.py`
- `rag_utils.py`
- `rag_pipeline.py`

Those are still high-coupling runtime orchestration modules.

## Cleanup Plan

Smells addressed:

1. **Root directory noise:** query planning, evaluation helpers, and persistence repository still live in the backend root.
2. **Boundary ambiguity:** query planning belongs with RAG; answer eval belongs with evaluation; conversation storage belongs with DB infrastructure.
3. **Compatibility risk:** tests and scripts import legacy names directly, so aliases must preserve module identity.

Order:

1. Run targeted behavior tests before moving.
2. Add package-boundary alias tests that fail before implementation.
3. Move implementations into their owning packages.
4. Add root `sys.modules` aliases.
5. Update internal imports to package paths.
6. Run targeted tests, then full gates.

## Verification Gates

Targeted behavior lock:

```powershell
uv run pytest tests/test_answer_eval.py tests/test_conversation_storage.py tests/test_query_plan_parser.py tests/test_document_scope_matching.py tests/test_filename_match_score.py tests/test_filename_normalization.py tests/test_heading_lexical_scoring.py tests/test_scoped_global_rrf.py tests/test_rag_utils.py -q
```

Post-move targeted gate:

```powershell
uv run pytest tests/test_backend_runtime_helper_boundaries.py tests/test_answer_eval.py tests/test_conversation_storage.py tests/test_query_plan_parser.py tests/test_document_scope_matching.py tests/test_filename_match_score.py tests/test_filename_normalization.py tests/test_heading_lexical_scoring.py tests/test_scoped_global_rrf.py tests/test_rag_utils.py -q
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
uv run pytest tests/test_answer_eval.py tests/test_conversation_storage.py tests/test_query_plan_parser.py tests/test_document_scope_matching.py tests/test_filename_match_score.py tests/test_filename_normalization.py tests/test_heading_lexical_scoring.py tests/test_scoped_global_rrf.py tests/test_rag_utils.py -q
96 passed, 1 warning
```

Expected failing boundary test before implementation:

```text
uv run pytest tests/test_backend_runtime_helper_boundaries.py -q
FAILED with ModuleNotFoundError: No module named 'evaluation'
```

Targeted post-move gate:

```text
uv run pytest tests/test_backend_runtime_helper_boundaries.py tests/test_answer_eval.py tests/test_conversation_storage.py tests/test_query_plan_parser.py tests/test_document_scope_matching.py tests/test_filename_match_score.py tests/test_filename_normalization.py tests/test_heading_lexical_scoring.py tests/test_scoped_global_rrf.py tests/test_rag_utils.py -q
97 passed, 1 warning
```
