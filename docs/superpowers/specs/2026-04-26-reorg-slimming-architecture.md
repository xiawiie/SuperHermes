# 2026-04-26 Reorg & Slimming Architecture Note

## What changed

This milestone applies compatibility-first slimming and structural preparation without changing external behavior.

### 1) Script-layer slimming

- Added shared utility module: `scripts/rag_eval/common.py`
- Moved generic helpers out of `scripts/evaluate_rag_matrix.py`:
  - list normalization
  - filename normalization set helpers
  - rate/percentile/presence helpers
- `scripts/evaluate_rag_matrix.py` keeps public behavior and CLI intact, but delegates helper logic to the shared module.

### 2) Backend modular scaffolding (compat-first)

- Added modular package skeleton:
  - `backend/application/main.py` (real FastAPI composition root)
  - `backend/routers/auth.py`
  - `backend/routers/chat.py`
  - `backend/routers/sessions.py`
  - `backend/routers/documents.py`
  - `backend/services/chat_service.py` (service facade for chat + stream)
  - `backend/services/document_service.py` (document list/upload/delete service)
  - `backend/rag/runtime/config.py` (centralized env parsing helpers)
- `backend/api.py` now acts as a router registry only.
- `backend/app.py` is a compatibility shim that re-exports `backend.application.main`.
- `backend/document_loader.py` reuses shared title/anchor rules from `backend/rag/rules.py`.

### 3) Frontend monolith slimming (safe split)

- Added:
  - `frontend/src/api.js`
  - `frontend/src/messages.js`
- Updated `frontend/index.html` to load helper modules before `frontend/script.js`.
- Updated `frontend/script.js` to use helper modules with fallback behavior retained.

## Compatibility mapping

- Existing imports/endpoints remain valid.
- New optional entrypoints are additive:
  - `backend.application.main:app`
  - `backend.services.chat_service:run_chat/run_chat_stream`
  - `backend.services.document_service:DocumentService`
  - `window.SuperHermesApi`
  - `window.SuperHermesMessages`

## 2026-04-26 RAG package convergence

The next convergence pass makes the existing `backend/rag` package the real home for focused RAG implementation modules. Root-level `rag_*.py` files now act as compatibility aliases only.

Moved implementation modules:

- `backend/rag/confidence.py`
- `backend/rag/context.py`
- `backend/rag/diagnostics.py`
- `backend/rag/profiles.py`
- `backend/rag/rerank.py`
- `backend/rag/retrieval.py`
- `backend/rag/trace.py`
- `backend/rag/types.py`

Legacy aliases kept for compatibility:

- `backend/rag_confidence.py`
- `backend/rag_context.py`
- `backend/rag_diagnostics.py`
- `backend/rag_profiles.py`
- `backend/rag_rerank.py`
- `backend/rag_retrieval.py`
- `backend/rag_trace.py`
- `backend/rag_types.py`

The aliases use `sys.modules[__name__] = packaged_module` so legacy imports and package imports share the same module object. This is intentional because existing tests and scripts patch module globals.

Deliberately not moved in this pass:

- `backend/rag_utils.py`: still a stateful compatibility facade.
- `backend/rag_pipeline.py`: still root-level because tests patch its module globals directly.

## 2026-04-26 shared/document package convergence

Pure helpers and document parsing were moved out of the backend root after adding package-boundary regression tests.

Moved implementation modules:

- `backend/shared/json_utils.py`
- `backend/shared/filename_normalization.py`
- `backend/documents/loader.py`

Legacy aliases kept for compatibility:

- `backend/json_utils.py`
- `backend/filename_normalization.py`
- `backend/document_loader.py`

Internal imports now prefer:

- `from backend.shared.json_utils import ...`
- `from backend.shared.filename_normalization import ...`
- `from backend.documents.loader import ...`

## 2026-04-26 infrastructure package convergence

Infrastructure adapters were moved out of the backend root after adding package-boundary regression tests.

Moved implementation modules:

- `backend/infra/cache.py`
- `backend/infra/embedding.py`
- `backend/infra/vector_store/milvus_client.py`
- `backend/infra/vector_store/milvus_writer.py`
- `backend/infra/vector_store/parent_chunk_store.py`

Legacy aliases kept for compatibility:

- `backend/cache.py`
- `backend/embedding.py`
- `backend/milvus_client.py`
- `backend/milvus_writer.py`
- `backend/parent_chunk_store.py`

Internal imports now prefer:

- `from backend.infra.cache import ...`
- `from backend.infra.embedding import ...`
- `from backend.infra.vector_store.milvus_client import ...`
- `from backend.infra.vector_store.milvus_writer import ...`
- `from backend.infra.vector_store.parent_chunk_store import ...`

## 2026-04-26 security/db/contracts convergence

Authentication, ORM/database setup, and Pydantic API contracts were moved out of the backend root after adding module-identity regression tests.

Moved implementation modules:

- `backend/security/auth.py`
- `backend/infra/db/database.py`
- `backend/infra/db/models.py`
- `backend/contracts/schemas.py`

Legacy aliases kept for compatibility:

- `backend/auth.py`
- `backend/database.py`
- `backend/models.py`
- `backend/schemas.py`

Internal imports now prefer:

- `from backend.security.auth import ...`
- `from backend.infra.db.database import ...`
- `from backend.infra.db.models import ...`
- `from backend.contracts.schemas import ...`

## 2026-04-26 runtime helper convergence

Query planning, answer evaluation, and conversation storage were moved into owning packages after adding module-identity regression tests.

Moved implementation modules:

- `backend/rag/query_plan.py`
- `backend/evaluation/answer_eval.py`
- `backend/infra/db/conversation_storage.py`

Legacy aliases kept for compatibility:

- `backend/query_plan.py`
- `backend/answer_eval.py`
- `backend/conversation_storage.py`

Internal imports now prefer:

- `from backend.rag.query_plan import ...`
- `from backend.evaluation.answer_eval import ...`
- `from backend.infra.db.conversation_storage import ...`

## 2026-04-26 chat runtime convergence

Chat agent runtime and LangChain tool bridge code were moved into the `backend/chat` package after adding app-startup and module-identity regression tests.

Moved implementation modules:

- `backend/chat/agent.py`
- `backend/chat/tools.py`

Legacy aliases kept for compatibility:

- `backend/agent.py`
- `backend/tools.py`

Internal imports now prefer:

- `from backend.chat.agent import ...`
- `from backend.chat.tools import ...`

## 2026-04-26 backend.rag.utils/pipeline convergence

The final high-risk backend.rag.utils and graph modules were moved after adding module-identity and monkeypatch-parity regression tests.

Moved implementation modules:

- `backend.rag.utils.py`
- `backend/rag/pipeline.py`

Legacy aliases kept for compatibility:

- `backend/rag_utils.py`
- `backend/rag_pipeline.py`

Internal imports now prefer:

- `from backend.rag.utils import ...`
- `from backend.rag.pipeline import ...`

## Rollback-friendly migration pattern

This milestone follows Step A (compatibility landing):

1. Add new module boundary without removing old path.
2. Route one call path through facade (`routers/* -> services/*`).
3. Keep old runtime behavior and test for parity.

Future Step B/C can continue with the same strategy:

- Step B: continue migrating internals in small slices.
- Step C: shrink compatibility exports once no references remain.
