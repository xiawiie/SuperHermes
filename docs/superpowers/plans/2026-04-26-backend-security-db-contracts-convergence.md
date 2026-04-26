# Backend Security, DB, and Contracts Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move authentication, database/ORM, and API schema implementations out of the backend root while preserving existing imports and runtime behavior.

**Architecture:** `backend/security/*` owns authentication and authorization helpers. `backend/infra/db/*` owns SQLAlchemy engine/session setup and ORM models. `backend/contracts/*` owns Pydantic request/response contracts. Root modules remain `sys.modules` compatibility aliases.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, Pydantic, pytest, ruff, compileall.

---

## Scope

Move implementations:

- `backend/auth.py` -> `backend/security/auth.py`
- `backend/database.py` -> `backend/infra/db/database.py`
- `backend/models.py` -> `backend/infra/db/models.py`
- `backend/schemas.py` -> `backend/contracts/schemas.py`

Keep compatibility aliases:

- `backend/auth.py`
- `backend/database.py`
- `backend/models.py`
- `backend/schemas.py`

Update internal imports in:

- `backend/application/main.py`
- `backend/conversation_storage.py`
- `backend/infra/vector_store/parent_chunk_store.py`
- `backend/routers/auth.py`
- `backend/routers/chat.py`
- `backend/routers/documents.py`
- `backend/routers/sessions.py`
- moved `security/auth.py`
- moved `infrastructure/db/models.py`
- moved `infrastructure/db/database.py`

Do not move:

- `agent.py`
- `tools.py`
- `answer_eval.py`
- `query_plan.py`
- `rag_utils.py`
- `rag_pipeline.py`

Those remain active runtime/facade modules and need their own smaller migration gates.

## Cleanup Plan

Smells addressed:

1. **Boundary violation:** auth, ORM, and API contracts still live in the backend root.
2. **Root directory noise:** root modules mix app entrypoints, compatibility aliases, and real implementation.
3. **Identity risk:** SQLAlchemy models and `Base` must stay identical across old and new imports.

Order:

1. Run targeted behavior tests before moving.
2. Add package-boundary alias tests that fail before implementation.
3. Move implementations into `security`, `infrastructure/db`, and `contracts`.
4. Add root `sys.modules` aliases.
5. Update internal imports to package paths.
6. Run targeted tests, then full gates.

## Verification Gates

Targeted behavior lock:

```powershell
uv run pytest tests/test_bootstrap.py tests/test_conversation_storage.py tests/test_parent_chunk_store_namespace.py tests/test_api_routes.py tests/test_application_entrypoints.py -q
```

Post-move targeted gate:

```powershell
uv run pytest tests/test_backend_security_db_contracts_boundaries.py tests/test_bootstrap.py tests/test_conversation_storage.py tests/test_parent_chunk_store_namespace.py tests/test_api_routes.py tests/test_application_entrypoints.py -q
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
uv run pytest tests/test_bootstrap.py tests/test_conversation_storage.py tests/test_parent_chunk_store_namespace.py tests/test_api_routes.py tests/test_application_entrypoints.py -q
10 passed
```

Expected failing boundary test before implementation:

```text
uv run pytest tests/test_backend_security_db_contracts_boundaries.py -q
FAILED with ModuleNotFoundError for security/infrastructure.db packages
```

Targeted post-move gate:

```text
uv run pytest tests/test_backend_security_db_contracts_boundaries.py tests/test_bootstrap.py tests/test_conversation_storage.py tests/test_parent_chunk_store_namespace.py tests/test_api_routes.py tests/test_application_entrypoints.py -q
12 passed
```
