# Backend Complete Migration Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the backend package migration with tests after every layer and a final anti-slop review.

**Architecture:** Follow `docs/superpowers/specs/2026-04-26-backend-complete-migration-design.md`. Move one layer at a time, keep root `sys.modules` aliases, update internal imports to package paths, and run targeted plus full verification gates.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, LangChain, LangGraph, pytest, ruff, Node smoke tests.

---

## Current Status

Completed:

- [x] Phase 1: router/service split
- [x] Phase 2: focused RAG modules
- [x] Phase 3: shared/documents
- [x] Phase 4: infrastructure
- [x] Phase 5: security/db/contracts
- [x] Phase 6: runtime helpers

Remaining:

- [x] Phase 7: chat runtime (`agent.py`, `tools.py`)
- [x] Phase 8: backend.rag.utils and pipeline (`rag_utils.py`, `rag_pipeline.py`)
- [x] Phase 9: compatibility alias audit
- [x] Phase 10: final review and slimming

## Phase 7: Chat Runtime

**Files:**

- Create: `backend/chat/__init__.py`
- Move: `backend/agent.py` -> `backend/chat/agent.py`
- Move: `backend/tools.py` -> `backend/chat/tools.py`
- Modify: `backend/agent.py` compatibility alias
- Modify: `backend/tools.py` compatibility alias
- Modify: `backend/chat/agent.py`
- Modify: `backend/services/chat_service.py`
- Modify: `backend/routers/sessions.py`
- Modify: `backend/rag_pipeline.py`
- Modify: `frontend/ui-redesign.test.mjs`
- Test: `tests/test_backend_chat_runtime_boundaries.py`

Steps:

- [ ] Run targeted pre-move gate.
- [ ] Add boundary test asserting `agent is chat.agent` and `tools is chat.tools`.
- [ ] Confirm boundary test fails before move.
- [ ] Move implementations into `backend/chat`.
- [ ] Add root compatibility aliases.
- [ ] Update internal imports to `chat.agent` and `chat.tools`.
- [ ] Run targeted post-move gate.
- [ ] Run full quality gates.

Commands:

```powershell
uv run pytest tests/test_bootstrap.py tests/test_api_routes.py tests/test_application_entrypoints.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py -q
uv run pytest tests/test_backend_chat_runtime_boundaries.py tests/test_bootstrap.py tests/test_api_routes.py tests/test_application_entrypoints.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py -q
```

## Phase 8: backend.rag.utils and Pipeline

**Files:**

- Create: `backend.rag.utils.py`
- Create: `backend/rag/pipeline.py`
- Move implementation from `backend/rag_utils.py` to `backend.rag.utils.py`
- Move implementation from `backend/rag_pipeline.py` to `backend/rag/pipeline.py`
- Modify: root `backend/rag_utils.py` compatibility alias
- Modify: root `backend/rag_pipeline.py` compatibility alias
- Modify imports that currently use `rag_utils` or `rag_pipeline` internally
- Test: `tests/test_backend.rag.utils_pipeline_boundaries.py`

Steps:

- [ ] Run targeted pre-move gate.
- [ ] Add boundary tests for module identity.
- [ ] Add monkeypatch parity checks for key globals patched by existing tests.
- [ ] Confirm boundary test fails before move.
- [ ] Move `rag_utils.py` implementation to `backend.rag.utils.py`.
- [ ] Move `rag_pipeline.py` implementation to `rag/pipeline.py`.
- [ ] Add root compatibility aliases.
- [ ] Update internal imports to `backend.rag.utils` and `rag.pipeline`.
- [ ] Run targeted post-move gate.
- [ ] Run full quality gates.

Commands:

```powershell
uv run pytest tests/test_rag_utils.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py tests/test_fallback_disabled_routing.py -q
uv run pytest tests/test_backend.rag.utils_pipeline_boundaries.py tests/test_rag_utils.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py tests/test_fallback_disabled_routing.py -q
```

## Phase 9: Compatibility Alias Audit

**Files:**

- Modify: `docs/ARCHITECTURE.md`
- Optional test additions if audit finds undocumented aliases.

Steps:

- [ ] List root `.py` files and classify each as entrypoint, compatibility alias, or implementation.
- [ ] Search internal imports for legacy root paths.
- [ ] Keep aliases that protect external or test compatibility.
- [ ] Document every remaining alias.
- [ ] Do not delete aliases unless tests and docs prove the path is no longer needed.

Commands:

```powershell
Get-ChildItem backend -File -Filter *.py | Select-Object Name,Length
Get-ChildItem -Path backend,scripts,tests -Recurse -File -Include *.py | Select-String -Pattern 'from rag_utils|import rag_utils|from rag_pipeline|import rag_pipeline|from agent|import agent|from tools|import tools'
```

## Phase 10: Final Review and Slimming

**Files:**

- Modify: docs only unless tests expose cleanup opportunities.

Steps:

- [ ] Run full tests.
- [ ] Run lint.
- [ ] Run compileall.
- [ ] Run frontend Node checks.
- [ ] Review root directory for implementation files.
- [ ] Review internal imports for legacy paths.
- [ ] Review docs for stale references to moved files.
- [ ] Update final acceptance doc with evidence.

Commands:

```powershell
uv run pytest tests/ -q
uv run ruff check backend/ scripts/ tests/
uv run python -m compileall backend scripts
node --check frontend\script.js
node --check frontend\src\api.js
node --check frontend\src\messages.js
node frontend\ui-redesign.test.mjs
```

## Final Execution Evidence

Root backend audit:

```text
All backend root implementation modules have been reduced to entrypoints or compatibility aliases.
Root compatibility aliases are 131-179 bytes each.
Entrypoints retained: app.py, api.py, __init__.py.
```

App import smoke:

```text
PYTHONPATH=backend uv run python -c "from application.main import app as app1; from app import app as app2; print(app1.title); print(len(app1.routes)); print(app1 is app2)"
SuperHermes Super Cute Pony Bot API
16
True
```

Final quality gates:

```text
uv run pytest tests/ -q
247 passed, 1 warning

uv run ruff check backend/ scripts/ tests/
All checks passed!

uv run python -m compileall backend scripts
success

node --check frontend\script.js
node --check frontend\src\api.js
node --check frontend\src\messages.js
node frontend\ui-redesign.test.mjs
all frontend checks passed
```
