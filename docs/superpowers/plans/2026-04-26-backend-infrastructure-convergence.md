# backend.infra Package Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move cache, embedding, Milvus, and parent-chunk infrastructure implementations out of the backend root while preserving existing imports and tests.

**Architecture:** `backend/infra/*` owns runtime infrastructure adapters. Root files remain `sys.modules` compatibility aliases so legacy import paths patch the same module objects as the packaged paths.

**Tech Stack:** Python 3.12, pytest, ruff, compileall.

---

## Scope

Move implementations:

- `backend/cache.py` -> `backend/infra/cache.py`
- `backend/embedding.py` -> `backend/infra/embedding.py`
- `backend/milvus_client.py` -> `backend/infra/vector_store/milvus_client.py`
- `backend/milvus_writer.py` -> `backend/infra/vector_store/milvus_writer.py`
- `backend/parent_chunk_store.py` -> `backend/infra/vector_store/parent_chunk_store.py`

Keep compatibility aliases:

- `backend/cache.py`
- `backend/embedding.py`
- `backend/milvus_client.py`
- `backend/milvus_writer.py`
- `backend/parent_chunk_store.py`

Update internal imports in:

- `backend/conversation_storage.py`
- `backend/rag_utils.py`
- `backend/services/document_service.py`
- moved infrastructure modules

Do not move `database.py` or `models.py` in this pass; the ORM boundary needs its own database-specific test gate.

## Cleanup Plan

Smells addressed:

1. **Root directory noise:** infrastructure adapters live beside app and domain entrypoints.
2. **Boundary ambiguity:** Milvus, Redis cache, embedding runtime, and parent chunk persistence are infrastructure concerns.
3. **Patch compatibility:** existing tests patch root module names, so aliases must preserve module identity.

Order:

1. Run targeted infrastructure tests before moving.
2. Add package-boundary alias tests that fail before implementation.
3. Move files into `infrastructure`.
4. Add root `sys.modules` aliases.
5. Update internal imports to package paths.
6. Run targeted tests, then full gates.

## Verification Gates

Targeted behavior lock:

```powershell
uv run pytest tests/test_bm25_state_isolation.py tests/test_milvus_client.py tests/test_milvus_index_version.py tests/test_parent_chunk_store_namespace.py tests/test_document_service.py tests/test_rag_utils.py -q
```

Post-move targeted gate:

```powershell
uv run pytest tests/test_backend.infra_boundaries.py tests/test_bm25_state_isolation.py tests/test_milvus_client.py tests/test_milvus_index_version.py tests/test_parent_chunk_store_namespace.py tests/test_document_service.py tests/test_rag_utils.py -q
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
uv run pytest tests/test_bm25_state_isolation.py tests/test_milvus_client.py tests/test_milvus_index_version.py tests/test_parent_chunk_store_namespace.py tests/test_document_service.py tests/test_rag_utils.py -q
35 passed, 1 warning
```

Expected failing boundary test before implementation:

```text
uv run pytest tests/test_backend.infra_boundaries.py -q
FAILED with ModuleNotFoundError: No module named 'infrastructure'
```

Targeted post-move gate:

```text
uv run pytest tests/test_backend.infra_boundaries.py tests/test_bm25_state_isolation.py tests/test_milvus_client.py tests/test_milvus_index_version.py tests/test_parent_chunk_store_namespace.py tests/test_document_service.py tests/test_rag_utils.py -q
36 passed, 1 warning
```
