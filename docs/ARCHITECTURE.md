# SuperHermes Architecture

## Overview

SuperHermes is a RAG (Retrieval-Augmented Generation) document assistant that provides accurate, evidence-grounded answers over technical manuals (network equipment, computing devices).

The system follows a **retrieval-first** philosophy: optimize retrieval quality before expanding to complex LLM capabilities.

## System Architecture

```
User Query
    │
    ▼
┌──────────┐    ┌──────────────┐
│  API     │───▶│  Agent       │
│  (api.py)│    │  (agent.py)  │
└──────────┘    └──────┬───────┘
                       │
              ┌────────▼────────┐
              │  RAG Pipeline   │
              │  (rag_pipeline) │
              └────────┬────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
┌─────────┐    ┌─────────────┐    ┌───────────┐
│QueryPlan│    │  Retrieval  │    │  Rerank   │
│         │───▶│  (retrieval)│───▶│  (rerank) │
└─────────┘    └──────┬──────┘    └─────┬─────┘
                      │                  │
                      ▼                  ▼
               ┌──────────┐       ┌───────────┐
               │ Milvus   │       │ Confidence│
               │ Client   │       │ Gate      │
               └──────────┘       └─────┬─────┘
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │   Context    │
                                  │  Assembly    │
                                  └──────┬───────┘
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │  LLM Answer  │
                                  │  Generation  │
                                  └──────────────┘
```

## Module Groups

### Core Application

| Module | Responsibility |
|--------|---------------|
| `app.py` | FastAPI application factory, CORS, lifespan |
| `api.py` | API routes: chat, documents, sessions |
| `auth.py` | JWT-based authentication |
| `schemas.py` | Pydantic request/response models |
| `models.py` | SQLAlchemy ORM models |
| `database.py` | Database connection and session management |

### RAG Pipeline

| Module | Responsibility |
|--------|---------------|
| `rag_pipeline.py` | LangGraph-based RAG orchestration graph |
| `rag_utils.py` | **Compatibility facade** — re-exports from sub-modules |
| `rag_retrieval.py` | Dense+sparse hybrid retrieval, doc scope, dedup |
| `rag_rerank.py` | Reranking with score fusion and pair enrichment |
| `rag_context.py` | Context window assembly and merge |
| `rag_confidence.py` | Retrieval confidence evaluation and fallback gate |
| `rag_trace.py` | RAG trace capture and diagnostics |
| `rag_types.py` | Shared type definitions |
| `rag_profiles.py` | Index profile normalization and chunk ID prefixing |
| `rag_diagnostics.py` | Retrieval diagnostics and miss analysis |

### LLM Integration

| Module | Responsibility |
|--------|---------------|
| `agent.py` | Chat agent with tool-augmented generation |
| `answer_eval.py` | LLM-based answer generation and evaluation |
| `query_plan.py` | QueryPlan extraction for scoped retrieval |
| `tools.py` | LangChain tool definitions (weather, RAG context) |

### Data Layer

| Module | Responsibility |
|--------|---------------|
| `document_loader.py` | Document parsing (PDF, DOCX, XLSX) and chunking |
| `embedding.py` | Embedding model management (BGE-M3) |
| `milvus_client.py` | Milvus vector store client with reconnection logic |
| `milvus_writer.py` | Milvus index writing and chunk management |
| `parent_chunk_store.py` | Parent chunk storage for context expansion |
| `filename_normalization.py` | Cross-platform filename normalization |
| `conversation_storage.py` | Conversation history (PostgreSQL) |
| `cache.py` | Redis caching layer |

## RAG Pipeline Detail

### Retrieval Flow

```
Query
  │
  ├─ QueryPlan ──▶ doc_scope (matched files)
  │
  ▼
Milvus hybrid search (dense + sparse RRF)
  │
  ├─ doc_scope boost (scoped candidates)
  ├─ heading_lexical boost
  ├─ global_reserve (fallback coverage)
  │
  ▼
Dedup (same_root_cap)
  │
  ▼
Rerank (cross-encoder with pair enrichment)
  │
  ├─ score fusion: rerank + rrf + scope + metadata
  │
  ▼
Confidence gate
  │
  ├─ fallback_required? ──▶ emit fallback signal
  │
  ▼
Context assembly (window merge, parent expansion)
  │
  ▼
LLM answer generation
```

### Index Profiles

| Profile | Collection | Text Mode | Description |
|---------|-----------|-----------|-------------|
| `gold_tc` | `embeddings_collection_gold_tc` | title_context | Gold standard title-context |
| `gold_tcf` | `embeddings_collection_gold_tcf` | title_context_filename | Gold standard TCF (GS3) |
| `v3_quality` | `embeddings_collection_v3_quality` | title_context_filename | V3Q quality ceiling |
| `v3_fast` | `embeddings_collection_v3_fast` | title_context_filename | V3F fast path (experimental) |

### Variant Taxonomy

| Tier | Variants | Role |
|------|----------|------|
| Deployable | GS3 | Default production baseline |
| Quality ceiling | V3Q | Slow but highest quality, fallback candidate |
| Experimental | V3F | Fast path (currently broken, needs rebuild) |
| Diagnostic | GS2HR | Strong file/root but chunk ID issues |

## Evaluation Framework

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/evaluate_rag_matrix.py` | Main evaluation orchestrator |
| `scripts/rag_eval/variants.py` | Variant configurations and pair definitions |
| `scripts/rag_eval/metrics.py` | Per-sample metric computation |
| `scripts/rag_eval/regression.py` | Saved-row summary regression |
| `scripts/review_rag_qrels.py` | Qrel review (pre-check, LLM, human queue) |
| `scripts/diagnose_variant_profile.py` | Variant index/collection health diagnosis |
| `scripts/align_rag_chunk_gold.py` | Page-level gold row alignment to chunk IDs |
| `scripts/rag_qrels.py` | Qrel matching, canonical IDs, merge logic |

### Metrics

| Metric | Description |
|--------|-------------|
| `File@5` | File hit in top 5 |
| `File+Page@5` | File+page hit in top 5 |
| `Chunk@5` | Canonical chunk hit in top 5 |
| `Root@5` | Root chunk hit in top 5 |
| `ChunkMRR` | Mean reciprocal rank of first chunk hit |
| `FileCandRecall` | File recall in full candidate set |
| `P50/P95 ms` | Retrieval latency percentiles |

### Qrel System

- **Qrel v2**: Initial alignment (87 aligned, 12 ambiguous, 26 failed)
- **Qrel v2.1**: LLM-reviewed (50 approved, 18 needs_review, coverage=0.856)
- Review states: `draft → llm_approved → approved` or `needs_human_review → corrected/rejected`

## Import Conventions

Backend modules use **bare imports** (`from rag_utils import ...`) because `backend/__init__.py` adds the backend directory to `sys.path`. This is intentional for compatibility with both `uvicorn backend.app:app` and in-directory execution.

**Facade pattern**: `rag_utils.py` re-exports functions from `rag_retrieval`, `rag_rerank`, `rag_context`, `rag_confidence`, `rag_trace`. Callers should import from `rag_utils` for backward compatibility.

## Environment Variables

Key environment variables (see `.env.example`):

| Variable | Purpose |
|----------|---------|
| `ARK_API_KEY` | LLM API key |
| `BASE_URL` | LLM API base URL |
| `MODEL` | Default LLM model |
| `GRADE_MODEL` | Evaluation/judge model |
| `MILVUS_HOST/PORT` | Milvus connection |
| `POSTGRES_*` | Database connection |
| `REDIS_*` | Cache connection |

## Testing

- **230 tests** in `tests/`
- Run: `uv run pytest tests/ -x -q`
- Lint: `uv run ruff check backend/ scripts/`

## Current Status (v3.1)

- GS3 is deployable baseline (File@5=0.984, P50=1144ms)
- V3Q is quality ceiling (File@5=0.992, P50=3938ms)
- V3F requires rebuild (File@5=0.416, profile/index mismatch suspected)
- Qrel v2.1 coverage: 0.856 (target: >=0.85, achieved)
