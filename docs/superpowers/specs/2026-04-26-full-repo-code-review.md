# 2026-04-26 Full Repo Code Review (Pre-refactor)

## Scope

Reviewed focus modules for correctness, maintainability, architecture fit, and refactor risk:

- `backend/rag_utils.py`
- `backend/rag_pipeline.py`
- `backend/document_loader.py`
- `backend/api.py`
- `scripts/evaluate_rag_matrix.py`
- `frontend/script.js`

## Findings

### [blocking] Import path mutation in runtime code

- File: `scripts/evaluate_rag_matrix.py`
- Evidence: inserts both project root and backend dir into `sys.path` at module import.
- Risk:
  - Runtime behavior depends on import order and process cwd.
  - Makes packaging and test isolation fragile.
- Action in refactor:
  - Move shared utilities to package imports only.
  - Keep CLI behavior unchanged but remove mutable `sys.path` dependency from evaluation core.

### [blocking] Global singleton initialization at import time

- Files:
  - `backend/rag_utils.py`
  - `backend/api.py`
- Evidence:
  - `rag_utils.py` initializes `MilvusManager` globally.
  - `api.py` initializes `DocumentLoader`, `ParentChunkStore`, `MilvusManager`, `MilvusWriter` globally.
- Risk:
  - Hidden side effects during import.
  - Harder to test, mock, and recover from runtime connection failures.
- Action in refactor:
  - Introduce lazy service factories and dependency-injected providers.
  - Keep old call paths as compatibility facade during migration.

### [important] Monolithic files with mixed responsibilities

- Files:
  - `backend/rag_utils.py` (config, retrieval orchestration, rerank, confidence, tracing)
  - `backend/rag_pipeline.py` (routing, fallback, grading, graph state, trace)
  - `scripts/evaluate_rag_matrix.py` (argparse + execution + reporting + regression)
  - `frontend/script.js` (auth, chat, upload, stream, UI state)
- Risk:
  - High cognitive load and larger change blast radius.
  - Small feature changes require touching high-risk files.
- Action in refactor:
  - Split by layer and responsibility while preserving external APIs.

### [important] Duplicate domain rules

- Files:
  - `backend/rag_utils.py`
  - `backend/document_loader.py`
- Evidence:
  - Similar anchor/title regex rules are duplicated.
- Risk:
  - Future tuning may drift across retrieval and ingestion paths.
- Action in refactor:
  - Centralize anchor/title normalization rules into shared module.

### [suggestion] API router is over-coupled to storage/index internals

- File: `backend/api.py`
- Evidence:
  - One router owns auth/session/chat/document indexing and directly touches low-level services.
- Risk:
  - Harder to reason about permission boundaries and failure handling.
- Action in refactor:
  - Split router modules (`auth`, `chat`, `sessions`, `documents`) and introduce thin service layer.

### [suggestion] Frontend behavior coupling

- File: `frontend/script.js`
- Evidence:
  - A single Vue app file handles auth, upload progress, SSE stream, history editing and UI rendering.
- Risk:
  - UI regressions more likely during changes.
- Action in refactor:
  - Split into `api`, `auth`, `chat`, `upload`, `view` modules, keep entry contract unchanged.

## Review Decision

- Result: **Request Changes**
- Reason:
  - The repository is functional, but current architecture has high refactor risk concentration in a few monolithic files.
  - Refactor should proceed in compatibility-first, test-gated slices (already reflected in execution plan).
