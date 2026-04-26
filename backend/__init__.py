"""
SuperHermes Backend — RAG-Augmented Document Assistant

Module Groups:
  Core Application:
    app           — FastAPI application factory
    api           — API route definitions
    auth          — Authentication and authorization
    schemas       — Pydantic request/response schemas
    models        — SQLAlchemy ORM models
    database      — Database connection management

  RAG Pipeline:
    rag_pipeline  — LangGraph-based RAG orchestration
    rag_utils     — Compatibility facade (re-exports from sub-modules)
    rag_retrieval — Dense+sparse hybrid retrieval, doc scope, dedup
    rag_rerank    — Reranking with score fusion and pair enrichment
    rag_context   — Context window assembly and merge
    rag_confidence— Retrieval confidence evaluation and fallback gate
    rag_trace     — RAG trace capture and diagnostics
    rag_types     — Shared type definitions
    rag_profiles  — Index profile normalization and chunk ID prefixing
    rag_diagnostics— Retrieval diagnostics and miss analysis

  LLM Integration:
    agent         — Chat agent with tool-augmented generation
    answer_eval   — LLM-based answer generation and evaluation
    query_plan    — QueryPlan extraction for scoped retrieval
    tools         — LangChain tool definitions

  Data Layer:
    document_loader — Document parsing and chunking
    embedding       — Embedding model management
    milvus_client   — Milvus vector store client
    milvus_writer   — Milvus index writing and management
    parent_chunk_store — Parent chunk storage for context expansion
    filename_normalization — Cross-platform filename normalization
    conversation_storage — Conversation history storage (PostgreSQL)
    cache           — Redis caching layer
"""

import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent
_backend_str = str(_backend_dir)
if _backend_str not in sys.path:
    sys.path.insert(0, _backend_str)
