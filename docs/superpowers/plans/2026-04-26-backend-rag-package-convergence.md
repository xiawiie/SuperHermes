# Backend RAG Package Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the real RAG implementation out of the backend root into `backend/rag/*` while preserving every legacy import and runtime behavior.

**Architecture:** This is plan B with a tighter boundary: reuse the existing `backend/rag` package as the stable RAG core instead of introducing a new `domain/rag` layer. Root-level `rag_*.py` modules become compatibility aliases only; `rag_utils.py` remains the public legacy facade until later passes can safely split it.

**Tech Stack:** Python 3.12, FastAPI backend, LangGraph RAG workflow, pytest, ruff, Node syntax smoke tests.

---

## Scope

This pass is intentionally limited to RAG structure convergence.

Move real implementations into:

- `backend/rag/confidence.py`
- `backend/rag/context.py`
- `backend/rag/diagnostics.py`
- `backend/rag/profiles.py`
- `backend/rag/rerank.py`
- `backend/rag/retrieval.py`
- `backend/rag/trace.py`
- `backend/rag/types.py`

Keep compatibility aliases at:

- `backend/rag_confidence.py`
- `backend/rag_context.py`
- `backend/rag_diagnostics.py`
- `backend/rag_profiles.py`
- `backend/rag_rerank.py`
- `backend/rag_retrieval.py`
- `backend/rag_trace.py`
- `backend/rag_types.py`

Do not move `backend/rag_pipeline.py` or `backend/rag_utils.py` in this pass. They are stateful and heavily monkeypatched by tests, so they stay as active compatibility/facade modules until their behavior has narrower coverage.

## Cleanup Plan

Smells addressed in this pass:

1. **Boundary violation:** RAG implementation files live in the backend root next to app entrypoints.
2. **Directory noise:** Root-level `rag_*.py` modules make the backend look flat and unowned.
3. **Compatibility risk:** Existing scripts/tests import legacy module names directly, so aliases must preserve module identity.

Order:

1. Add a regression test that requires legacy modules to alias the new package modules.
2. Move focused RAG modules into `backend/rag`.
3. Replace legacy modules with `sys.modules` aliases, not `import *`, so monkeypatching and module globals keep working.
4. Update internal imports in `rag_utils.py`, `rag_pipeline.py`, `milvus_writer.py`, and `parent_chunk_store.py` to the package paths.
5. Update architecture docs to show root files as compatibility aliases.
6. Run targeted tests after the move, then full gates.

## Verification Gates

Run before code movement:

```powershell
uv run pytest tests/test_rag_utils.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_trace.py -q
```

Run after the package move:

```powershell
uv run pytest tests/test_backend_rag_package_boundaries.py tests/test_rag_utils.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_trace.py tests/test_rag_diagnostics.py -q
```

Run final gates:

```powershell
uv run pytest tests/ -q
uv run ruff check backend/ scripts/ tests/
uv run python -m compileall backend scripts
node --check frontend\script.js
node --check frontend\src\api.js
node --check frontend\src\messages.js
node frontend\ui-redesign.test.mjs
```

## Tasks

### Task 1: Lock RAG Package Boundary

**Files:**

- Create: `tests/test_backend_rag_package_boundaries.py`

- [x] Add a test that imports each legacy module and each new package module.
- [x] Assert `importlib.import_module("rag_retrieval") is importlib.import_module("rag.retrieval")`.
- [x] Repeat for confidence, context, diagnostics, profiles, rerank, trace, and types.
- [x] Run the test before implementation and confirm it fails because the package modules do not exist yet.

### Task 2: Move Focused RAG Modules

**Files:**

- Move: `backend/rag_confidence.py` -> `backend/rag/confidence.py`
- Move: `backend/rag_context.py` -> `backend/rag/context.py`
- Move: `backend/rag_diagnostics.py` -> `backend/rag/diagnostics.py`
- Move: `backend/rag_profiles.py` -> `backend/rag/profiles.py`
- Move: `backend/rag_rerank.py` -> `backend/rag/rerank.py`
- Move: `backend/rag_retrieval.py` -> `backend/rag/retrieval.py`
- Move: `backend/rag_trace.py` -> `backend/rag/trace.py`
- Move: `backend/rag_types.py` -> `backend/rag/types.py`

- [x] Preserve file contents during moves.
- [x] Keep `backend/rag/rules.py` and `backend/rag/runtime/config.py` where they are.

### Task 3: Add Legacy Module Aliases

**Files:**

- Modify: root `backend/rag_*.py` compatibility files

Each legacy file should contain only:

```python
"""Compatibility alias for the packaged RAG module."""

import sys

from rag import retrieval as _module

sys.modules[__name__] = _module
```

Use the matching package module in each file.

### Task 4: Update Internal Imports

**Files:**

- Modify: `backend/rag_utils.py`
- Modify: `backend/rag_pipeline.py`
- Modify: `backend/milvus_writer.py`
- Modify: `backend/parent_chunk_store.py`
- Modify: `frontend/ui-redesign.test.mjs`

Replace internal imports such as `from rag_retrieval import ...` with `from backend.rag.retrieval import ...`. Keep external compatibility imports untouched in tests and scripts unless a test intentionally checks the new boundary.

### Task 5: Document Architecture State

**Files:**

- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/superpowers/specs/2026-04-26-reorg-slimming-architecture.md`

Document:

- `backend/rag/*` is the RAG implementation package.
- root `rag_*.py` files are legacy aliases.
- `rag_utils.py` remains the compatibility facade and is deliberately not moved in this pass.
- remaining follow-up: split `rag_utils.py` after behavior-specific tests are strong enough.

### Task 6: Verify

- [x] Run the targeted RAG test gate.
- [x] Fix import or alias regressions if any.
- [x] Run full pytest.
- [x] Run ruff.
- [x] Run compileall.
- [x] Run frontend Node syntax checks and UI smoke test.

## Execution Evidence

Pre-change behavior lock:

```text
uv run pytest tests/test_rag_utils.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_trace.py -q
27 passed, 1 warning
```

Expected failing boundary test before implementation:

```text
uv run pytest tests/test_backend_rag_package_boundaries.py -q
FAILED with ModuleNotFoundError: No module named 'rag.confidence'
```

Targeted post-move RAG gate:

```text
uv run pytest tests/test_backend_rag_package_boundaries.py tests/test_rag_utils.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_trace.py tests/test_rag_diagnostics.py -q
35 passed, 1 warning
```

Final gates:

```text
uv run pytest tests/ -q
237 passed, 1 warning

uv run ruff check backend/ scripts/ tests/
All checks passed!

uv run python -m compileall backend scripts
success

node --check frontend\script.js
node --check frontend\src\api.js
node --check frontend\src\messages.js
node frontend\ui-redesign.test.mjs
all frontend smoke checks passed
```
