# Backend Chat Runtime Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move chat agent and LangChain tool implementations out of the backend root while preserving legacy imports and app startup behavior.

**Architecture:** `backend/chat/agent.py` owns chat agent orchestration. `backend/chat/tools.py` owns LangChain tool definitions and RAG step/context bridge state. Root `agent.py` and `tools.py` remain compatibility aliases.

**Tech Stack:** Python 3.12, LangChain, FastAPI, pytest, ruff, compileall, Node smoke tests.

---

## Scope

Move implementations:

- `backend/agent.py` -> `backend/chat/agent.py`
- `backend/tools.py` -> `backend/chat/tools.py`

Keep compatibility aliases:

- `backend/agent.py`
- `backend/tools.py`

Update internal imports:

- `backend/chat/agent.py`
- `backend/services/chat_service.py`
- `backend/routers/sessions.py`
- `backend/rag_pipeline.py`
- `frontend/ui-redesign.test.mjs`

Do not move:

- `rag_utils.py`
- `rag_pipeline.py`

Those stay as final backend.rag.utils/orchestration migration candidates.

## Cleanup Plan

Smells addressed:

1. **Root directory noise:** chat agent and tool runtime still live beside entrypoint aliases.
2. **Boundary ambiguity:** chat runtime should be grouped separately from RAG internals and infrastructure.
3. **Startup risk:** app import must still avoid eager LLM/embedding initialization.

Order:

1. Run targeted startup/chat/RAG tests before moving.
2. Add package-boundary alias tests that fail before implementation.
3. Move implementations into `backend/chat`.
4. Add root `sys.modules` aliases.
5. Update internal imports to package paths.
6. Run targeted tests, then full gates.

## Verification Gates

Targeted behavior lock:

```powershell
uv run pytest tests/test_bootstrap.py tests/test_api_routes.py tests/test_application_entrypoints.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py -q
```

Post-move targeted gate:

```powershell
uv run pytest tests/test_backend_chat_runtime_boundaries.py tests/test_bootstrap.py tests/test_api_routes.py tests/test_application_entrypoints.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py -q
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
uv run pytest tests/test_bootstrap.py tests/test_api_routes.py tests/test_application_entrypoints.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py -q
14 passed, 1 warning
```

Expected failing boundary test before implementation:

```text
uv run pytest tests/test_backend_chat_runtime_boundaries.py -q
FAILED with ModuleNotFoundError: No module named 'chat'
```

Targeted post-move gate:

```text
uv run pytest tests/test_backend_chat_runtime_boundaries.py tests/test_bootstrap.py tests/test_api_routes.py tests/test_application_entrypoints.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py -q
15 passed, 1 warning
```
