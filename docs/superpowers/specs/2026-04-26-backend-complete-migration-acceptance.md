# Backend Canonical Import Acceptance

## Summary

The backend migration now uses canonical `backend.*` imports only. Legacy bare imports are intentionally unsupported, and root alias modules have been deleted.

## Final Package Layout

```text
backend/
  __init__.py
  api.py
  app.py
  application/
  chat/
  contracts/
  data/
  documents/
  evaluation/
  infra/
    db/
    vector_store/
  rag/
    runtime/
  routers/
  security/
  services/
  shared/
```

## Import Contract

Supported:

- `from backend.rag.utils import ...`
- `from backend.rag.pipeline import ...`
- `from backend.infra.db.database import ...`
- `from backend.infra.vector_store.milvus_client import ...`
- `from backend.documents.loader import ...`
- `from backend.evaluation.answer_eval import ...`

Unsupported:

- `import rag_utils`
- `import rag_pipeline`
- `import database`
- `import document_loader`
- `PYTHONPATH=backend python -c "import rag_utils"`

## Root Directory Audit

Allowed root Python files:

- `backend/__init__.py`
- `backend/api.py`
- `backend/app.py`

Forbidden root alias files are covered by `tests/test_import_contract.py`.

## Entrypoints

Supported:

```powershell
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
uv run python backend/app.py
```

`backend/app.py` contains the only script-mode bootstrap needed for direct execution. `backend/__init__.py` is package-only.

## Verification Plan

Full gate:

```powershell
uv run pytest tests/ -q
uv run ruff check backend/ scripts/ tests/
uv run python -m compileall backend scripts
node --check frontend\script.js
node --check frontend\src\api.js
node --check frontend\src\messages.js
node frontend\ui-redesign.test.mjs
```

RAG smoke:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile smoke --variants B0 --skip-reindex --limit 1 --run-id post-cleanup-smoke
uv run python scripts\evaluate_rag_matrix.py --dataset-profile smoke --variants B0 --mode graph --skip-reindex --limit 1 --run-id post-cleanup-graph-smoke
```
