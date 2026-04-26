# SuperHermes Architecture

## Overview

SuperHermes is a FastAPI + RAG document assistant. The backend is organized as a Python package with canonical `backend.*` imports. Legacy bare imports such as `import rag_utils`, `import database`, and `from document_loader import ...` are intentionally unsupported.

The backend root now contains only entrypoints, package metadata, data, and real package directories. Former root alias modules have been deleted.

## Runtime Flow

```text
Frontend
  -> backend.app
  -> backend.application.main
  -> backend.api
  -> backend.routers.*
  -> backend.services.*
  -> backend.chat / backend.rag
  -> backend.infra
  -> PostgreSQL, Redis, Milvus
```

RAG answer flow:

```text
User question
  -> backend.chat.agent
  -> backend.chat.tools
  -> backend.rag.pipeline
  -> backend.rag.utils
  -> backend.rag retrieval/rerank/context/confidence modules
  -> backend.infra.vector_store and backend.infra.db
```

## Backend Root Contract

Allowed root Python files:

```text
backend/__init__.py
backend/api.py
backend/app.py
```

Allowed root package directories:

```text
backend/application/
backend/chat/
backend/contracts/
backend/data/
backend/documents/
backend/evaluation/
backend/infra/
backend/rag/
backend/routers/
backend/security/
backend/services/
backend/shared/
```

`backend/__init__.py` is package-only and must not mutate `sys.path` or register `sys.modules` aliases.

## Application And HTTP

| Path | Responsibility |
| --- | --- |
| `backend/app.py` | ASGI/script entrypoint; contains the only script-mode bootstrap for `python backend/app.py` |
| `backend/application/main.py` | FastAPI app factory, lifespan, CORS, static mount |
| `backend/api.py` | Router registry only |
| `backend/routers/auth.py` | Auth HTTP routes |
| `backend/routers/chat.py` | Chat and streaming HTTP routes |
| `backend/routers/sessions.py` | Session HTTP routes |
| `backend/routers/documents.py` | Document HTTP routes |
| `backend/services/chat_service.py` | Chat use-case facade |
| `backend/services/document_service.py` | Document list/upload/delete orchestration |
| `backend/contracts/schemas.py` | Pydantic request/response contracts |
| `backend/security/auth.py` | Authentication, password hashing, JWT, authorization dependencies |

Rules:

- Routers own HTTP request/response concerns.
- Services own use-case orchestration.
- `backend/api.py` stays a router registry and does not hold business logic.
- Entry compatibility belongs in `backend/app.py`, not `backend/__init__.py`.

## RAG Core

| Path | Responsibility |
| --- | --- |
| `backend/rag/utils.py` | RAG helper/facade behavior and retrieval orchestration |
| `backend/rag/pipeline.py` | LangGraph RAG workflow |
| `backend/rag/query_plan.py` | Query planning, document-scope matching, route selection |
| `backend/rag/retrieval.py` | Dense+sparse retrieval helpers, filename boost, dedupe |
| `backend/rag/rerank.py` | Rerank provider calls, pair enrichment, score fusion |
| `backend/rag/context.py` | Context assembly, parent merge, structure/root rerank |
| `backend/rag/confidence.py` | Retrieval confidence gate and anchor matching |
| `backend/rag/diagnostics.py` | Retrieval failure classification and diagnostics |
| `backend/rag/profiles.py` | Index profile normalization and chunk ID prefixing |
| `backend/rag/rules.py` | Shared heading/title/anchor parsing rules |
| `backend/rag/trace.py` | Candidate identity, text hashing, golden trace signatures |
| `backend/rag/types.py` | Shared RAG trace/error types |
| `backend/rag/runtime/config.py` | Environment parsing helpers |

## Data And Infra

| Path | Responsibility |
| --- | --- |
| `backend/infra/cache.py` | Redis cache wrapper |
| `backend/infra/embedding.py` | Embedding model/runtime management |
| `backend/infra/db/database.py` | Database engine/session setup |
| `backend/infra/db/models.py` | SQLAlchemy ORM models |
| `backend/infra/db/conversation_storage.py` | Chat history persistence and read-through cache |
| `backend/infra/vector_store/milvus_client.py` | Milvus client and search operations |
| `backend/infra/vector_store/milvus_writer.py` | Milvus write/index orchestration |
| `backend/infra/vector_store/parent_chunk_store.py` | Parent chunk storage for context expansion |

## Documents And Shared Helpers

| Path | Responsibility |
| --- | --- |
| `backend/documents/loader.py` | Document parsing and chunking |
| `backend/shared/filename_normalization.py` | Filename normalization |
| `backend/shared/json_utils.py` | JSON extraction helpers |
| `backend/evaluation/answer_eval.py` | Offline answer generation/evaluation helpers |

## Import Policy

Supported:

```python
from backend.rag.utils import retrieve_documents
from backend.rag.pipeline import run_rag_graph
from backend.infra.db.database import init_db
from backend.infra.vector_store.milvus_client import MilvusManager
from backend.documents.loader import DocumentLoader
```

Unsupported:

```python
import rag_utils
import rag_pipeline
import database
from document_loader import DocumentLoader
```

Also unsupported:

```powershell
PYTHONPATH=backend python -c "import rag_utils"
```

## Supported Entrypoints

```powershell
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
uv run python backend/app.py
```

`python backend/app.py` is entrypoint compatibility only. It does not imply support for legacy root imports.

## Verification Gates

Standard gates for architecture cleanup:

```powershell
uv run pytest tests/ -q
uv run ruff check backend/ scripts/ tests/
uv run python -m compileall backend scripts
node --check frontend\script.js
node --check frontend\src\api.js
node --check frontend\src\messages.js
node frontend\ui-redesign.test.mjs
```

RAG performance smoke:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile smoke --variants B0 --skip-reindex --limit 1 --run-id post-cleanup-smoke
uv run python scripts\evaluate_rag_matrix.py --dataset-profile smoke --variants B0 --mode graph --skip-reindex --limit 1 --run-id post-cleanup-graph-smoke
```
