# Backend Router Reorg Stage 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the backend HTTP layer into clear router modules so the app entrypoint becomes a thin composition root, while preserving every current endpoint and test contract.

**Architecture:** `backend/application/main.py` owns FastAPI creation and middleware. `backend/api.py` becomes a router registry only. Capability routers live in `backend/routers/{auth,chat,sessions,documents}.py`, and business logic stays in `backend/services/*`. Existing flat modules remain as compatibility dependencies only where they still carry core domain logic.

**Tech Stack:** FastAPI, pytest, Ruff, Node for the frontend contract test.

---

### Task 1: Lock route registration before moving handlers

**Files:**
- Create: `tests/test_api_routes.py`
- Modify: `frontend/ui-redesign.test.mjs`

- [ ] **Step 1: Write the failing test**

Add a Python test that imports `backend.app:create_app()` and asserts the app exposes these paths: `/auth/register`, `/auth/login`, `/auth/me`, `/chat`, `/chat/stream`, `/sessions`, `/sessions/{session_id}`, `/documents`, `/documents/upload`, `/documents/{filename}`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_routes.py -q`
Expected: fail until the new router modules are wired into the app.

- [ ] **Step 3: Update the frontend contract test to look at router modules**

Change `frontend/ui-redesign.test.mjs` so the backend string checks read `backend/routers/chat.py`, `backend/routers/auth.py`, `backend/routers/sessions.py`, `backend/routers/documents.py`, and `backend/api.py` only for router registration.

- [ ] **Step 4: Run the test again**

Run: `node frontend/ui-redesign.test.mjs`
Expected: fail until the route handlers are moved.

### Task 2: Split the HTTP handlers into router modules

**Files:**
- Create: `backend/routers/auth.py`
- Create: `backend/routers/sessions.py`
- Create: `backend/routers/documents.py`
- Modify: `backend/routers/chat.py`
- Modify: `backend/api.py`

- [ ] **Step 1: Move auth/session/document/chat handlers out of `backend/api.py`**

`backend/routers/auth.py` owns register/login/me.

`backend/routers/sessions.py` owns session list/get/rename/delete.

`backend/routers/documents.py` owns list/upload/delete and uses `get_document_service()`.

`backend/routers/chat.py` owns `/chat` and `/chat/stream` and keeps the current error translation.

- [ ] **Step 2: Reduce `backend/api.py` to router composition**

Import the four routers and `include_router()` each one. Keep no endpoint bodies there.

- [ ] **Step 3: Run the targeted regression tests**

Run: `uv run pytest tests/test_bootstrap.py tests/test_document_service.py tests/test_api_routes.py -q`
Expected: PASS after the router split.

### Task 3: Make the app entrypoint the real composition root

**Files:**
- Modify: `backend/application/main.py`
- Modify: `backend/app.py`
- Modify: `backend/__init__.py` only if import behavior needs to stay compatible

- [ ] **Step 1: Move app factory logic into `backend/application/main.py`**

Keep CORS, no-cache middleware, `init_db()` startup, and `StaticFiles` mounting there.

- [ ] **Step 2: Turn `backend/app.py` into a thin compatibility shim**

It should only import `app` and `create_app` from `backend.application.main`.

- [ ] **Step 3: Run import smoke tests**

Run: `uv run python -c "from backend.app import app; print(app.title)"` and `uv run python -c "from backend.application.main import app; print(app.title)"`
Expected: both print the same app title.

### Task 4: Reconcile docs and compatibility surfaces

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/superpowers/specs/2026-04-26-reorg-slimming-architecture.md`
- Modify: `docs/superpowers/specs/2026-04-26-refactor-final-acceptance.md`

- [ ] **Step 1: Update the architecture map**

Document the new layer order: `application -> api/router registry -> routers -> services -> domain/infrastructure`.

- [ ] **Step 2: Update the acceptance notes**

Record the router split, the thin app entrypoint, and the new test count after the route split.

- [ ] **Step 3: Run docs-adjacent checks**

Run: `uv run ruff check backend/ scripts/ tests/` and `node frontend/ui-redesign.test.mjs`
Expected: PASS.

### Task 5: Final whole-repo verification

**Files:**
- No new files expected

- [ ] **Step 1: Run the full Python and frontend checks**

Run: `uv run pytest tests/ -q`
Run: `uv run python -m compileall backend scripts`
Run: `uv run ruff check backend/ scripts/ tests/`
Run: `node --check frontend/script.js`
Run: `node --check frontend/src/api.js`
Run: `node --check frontend/src/messages.js`
Run: `node frontend/ui-redesign.test.mjs`

- [ ] **Step 2: Review for dead compatibility layers**

Check `backend/routers/*.py`, `backend/application/main.py`, and `backend/api.py` for any leftover duplicate route bodies or pass-through wrappers that can be deleted in a follow-up pass.

