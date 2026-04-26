# Backend Complete Migration Design

## Intent

This design completes the SuperHermes backend migration without changing API, CLI, script, or frontend behavior. The goal is to turn the backend root from a mixed implementation directory into a compatibility surface, with real code grouped by responsibility.

The migration uses compatibility-first steps:

1. Lock behavior with targeted tests.
2. Add package-boundary tests that fail before each move.
3. Move one layer at a time.
4. Keep root-level `sys.modules` aliases until old imports are intentionally retired.
5. Run targeted tests after each layer.
6. Run full quality gates after each meaningful batch.

## Design Options

### Option A: Stop at the Current State

Keep root aliases and do no more structural movement.

Trade-off: lowest risk, but leaves `agent.py`, `tools.py`, `rag_utils.py`, and `rag_pipeline.py` as visible root implementation files.

### Option B: Compatibility-First Full Convergence

Continue moving implementation modules into owning packages while keeping root compatibility aliases.

Trade-off: more files move, but each step is reversible and testable. This is the chosen option.

### Option C: Remove All Root Compatibility Files Now

Move everything and delete old import paths immediately.

Trade-off: cleanest tree visually, but high risk because tests, scripts, local workflows, and possible external commands still use bare imports.

## Target Backend Layout

```text
backend/
  application/
    main.py
  routers/
    auth.py
    chat.py
    documents.py
    sessions.py
  services/
    chat_service.py
    document_service.py
  contracts/
    schemas.py
  security/
    auth.py
  chat/
    agent.py
    tools.py
  rag/
    pipeline.py
    facade.py
    query_plan.py
    retrieval.py
    rerank.py
    context.py
    confidence.py
    diagnostics.py
    profiles.py
    rules.py
    trace.py
    types.py
    runtime/config.py
  documents/
    loader.py
  infrastructure/
    cache.py
    embedding.py
    db/
      database.py
      models.py
      conversation_storage.py
    vector_store/
      milvus_client.py
      milvus_writer.py
      parent_chunk_store.py
  evaluation/
    answer_eval.py
  shared/
    filename_normalization.py
    json_utils.py
  app.py
  api.py
  *.py compatibility aliases
```

Root files that should remain by design:

- `app.py`: legacy ASGI entrypoint.
- `api.py`: router registry.
- Compatibility aliases for old import paths until the repository and external commands no longer need them.
- `__init__.py`: current bare-import compatibility hook. Remove only in a later import-policy migration.

## Layer Ownership

### Application Layer

Owns FastAPI assembly only.

- `application/main.py`: app factory, lifespan, middleware, static mount.
- `api.py`: includes routers only.
- `app.py`: legacy shim.

No business logic should be added here.

### Router Layer

Owns HTTP protocol details only.

- request models
- dependencies
- status codes
- `HTTPException`

Routers call services or lower-level dependencies; they do not perform ingestion, retrieval, or persistence orchestration.

### Service Layer

Owns user-facing use cases.

- `chat_service.py`: chat and streaming facade.
- `document_service.py`: list/upload/delete document workflows.

Services may call chat runtime, documents, infrastructure, and RAG modules.

### Contracts Layer

Owns Pydantic request/response models.

- `contracts/schemas.py`

Routers import from `contracts.schemas`. Legacy `schemas.py` remains an alias.

### Security Layer

Owns auth and authorization.

- password hashing
- JWT encode/decode
- FastAPI auth dependencies
- role resolution

Legacy `auth.py` remains an alias.

### Chat Runtime Layer

Owns the LangChain agent and tool bridge.

- `chat/agent.py`
- `chat/tools.py`

This layer must stay lazy enough that importing `app.py` does not initialize remote LLM or embedding dependencies.

### RAG Layer

Owns retrieval and RAG workflow internals.

- focused modules already live under `rag/*`
- `rag/query_plan.py` owns document-scope parsing
- `backend.rag.utils.py` should eventually own the current `rag_utils.py` public helper/facade behavior
- `rag/pipeline.py` should eventually own the current `rag_pipeline.py` graph

`rag_utils.py` and `rag_pipeline.py` are last because tests patch module-level state directly.

### Documents Layer

Owns source document parsing and chunk preparation.

- `documents/loader.py`

### Infrastructure Layer

Owns external system adapters.

- DB
- cache
- embedding runtime
- Milvus/vector store
- parent chunk store

Infrastructure modules should not import backend.routers.

### Evaluation Layer

Owns offline answer evaluation helpers.

- `evaluation/answer_eval.py`
- `scripts/rag_eval/*`

## Migration Phases

### Phase 0: Baseline Gate

Run:

```powershell
uv run pytest tests/ -q
uv run ruff check backend/ scripts/ tests/
uv run python -m compileall backend scripts
node --check frontend\script.js
node --check frontend\src\api.js
node --check frontend\src\messages.js
node frontend\ui-redesign.test.mjs
```

### Phase 1: Router and Service Split

Status: complete.

Files:

- `application/main.py`
- `routers/*`
- `services/*`

Gate:

```powershell
uv run pytest tests/test_api_routes.py tests/test_application_entrypoints.py tests/test_document_service.py -q
```

### Phase 2: Focused RAG Modules

Status: complete.

Files:

- `rag/retrieval.py`
- `rag/rerank.py`
- `rag/context.py`
- `rag/confidence.py`
- `rag/diagnostics.py`
- `rag/profiles.py`
- `rag/trace.py`
- `rag/types.py`

Gate:

```powershell
uv run pytest tests/test_backend_rag_package_boundaries.py tests/test_rag_utils.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_trace.py tests/test_rag_diagnostics.py -q
```

### Phase 3: Shared and Documents

Status: complete.

Files:

- `shared/json_utils.py`
- `shared/filename_normalization.py`
- `documents/loader.py`

Gate:

```powershell
uv run pytest tests/test_backend_shared_documents_boundaries.py tests/test_answer_eval.py tests/test_filename_normalization.py tests/test_document_loader.py tests/test_document_service.py tests/test_query_plan_parser.py tests/test_rag_utils.py -q
```

### Phase 4: Infrastructure

Status: complete.

Files:

- `infrastructure/cache.py`
- `infrastructure/embedding.py`
- `infrastructure/vector_store/*`

Gate:

```powershell
uv run pytest tests/test_backend.infra_boundaries.py tests/test_bm25_state_isolation.py tests/test_milvus_client.py tests/test_milvus_index_version.py tests/test_parent_chunk_store_namespace.py tests/test_document_service.py tests/test_rag_utils.py -q
```

### Phase 5: Security, DB, and Contracts

Status: complete.

Files:

- `security/auth.py`
- `infrastructure/db/database.py`
- `infrastructure/db/models.py`
- `contracts/schemas.py`

Gate:

```powershell
uv run pytest tests/test_backend_security_db_contracts_boundaries.py tests/test_bootstrap.py tests/test_conversation_storage.py tests/test_parent_chunk_store_namespace.py tests/test_api_routes.py tests/test_application_entrypoints.py -q
```

### Phase 6: Runtime Helpers

Status: complete.

Files:

- `rag/query_plan.py`
- `evaluation/answer_eval.py`
- `infrastructure/db/conversation_storage.py`

Gate:

```powershell
uv run pytest tests/test_backend_runtime_helper_boundaries.py tests/test_answer_eval.py tests/test_conversation_storage.py tests/test_query_plan_parser.py tests/test_document_scope_matching.py tests/test_filename_match_score.py tests/test_filename_normalization.py tests/test_heading_lexical_scoring.py tests/test_scoped_global_rrf.py tests/test_rag_utils.py -q
```

### Phase 7: Chat Runtime

Status: next.

Files:

- `chat/agent.py`
- `chat/tools.py`

Gate:

```powershell
uv run pytest tests/test_backend_chat_runtime_boundaries.py tests/test_bootstrap.py tests/test_api_routes.py tests/test_application_entrypoints.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py -q
```

### Phase 8: backend.rag.utils and Pipeline

Status: final high-risk migration.

Target:

- `backend.rag.utils.py` for current `rag_utils.py`
- `rag/pipeline.py` for current `rag_pipeline.py`

Constraints:

- Existing tests patch `rag_pipeline` globals directly.
- `rag_utils.py` owns environment constants, global Milvus manager, parent store cache, rerank runtime, retrieval orchestration, and helper facade behavior.

Required test additions:

- alias identity for `rag_utils -> backend.rag.utils`
- alias identity for `rag_pipeline -> rag.pipeline`
- monkeypatch parity tests for key module globals
- retrieval metadata regression around `stage_errors`, timing fields, `context_files`, `fallback_required`

Gate:

```powershell
uv run pytest tests/test_backend.rag.utils_pipeline_boundaries.py tests/test_rag_utils.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py tests/test_fallback_disabled_routing.py -q
```

### Phase 9: Compatibility Alias Audit

Do not delete root aliases until all conditions are met:

1. Repository code no longer imports the legacy module internally.
2. Tests prove package and legacy modules are identical when both exist.
3. Documentation lists the alias as intentionally retained or safe to remove.
4. External entrypoint risk is understood.

Aliases may remain if they protect common commands such as:

```powershell
PYTHONPATH=backend python -c "import app"
PYTHONPATH=backend python -c "import rag_utils"
```

### Phase 10: Final Review and Slimming

Run a final anti-slop pass:

- dead code scan
- duplicate helper scan
- import-policy scan
- root-directory real-implementation scan
- test coverage scan
- docs consistency scan

Final gate:

```powershell
uv run pytest tests/ -q
uv run ruff check backend/ scripts/ tests/
uv run python -m compileall backend scripts
node --check frontend\script.js
node --check frontend\src\api.js
node --check frontend\src\messages.js
node frontend\ui-redesign.test.mjs
```

## Error Handling and Rollback

Each phase is rollback-friendly:

- moved implementation files can be moved back
- root aliases preserve old imports
- failing tests identify the layer that broke
- no phase introduces a new dependency

If a targeted gate fails:

1. Stop the next migration phase.
2. Fix the current phase only.
3. Re-run the targeted gate.
4. Re-run full gates before continuing.

## Completion Criteria

The migration is complete when:

- root-level implementation files are limited to entrypoints, compatibility aliases, and explicitly documented high-level facades
- internal imports prefer package paths
- architecture docs match the actual code tree
- all targeted boundary tests pass
- all full quality gates pass
- remaining compatibility files are documented as intentional, not forgotten leftovers
