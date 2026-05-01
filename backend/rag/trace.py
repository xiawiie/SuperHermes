from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, MutableMapping
from typing import Any

from backend.rag.types import RetrievalMeta, StageError, StageErrorDict

TRACE_SIGNATURE_VERSION = "rag-trace-signature-v1"
RAG_CONTEXT_FORMAT_VERSION = "retrieved-chunks-v1"
STANDARD_EXECUTION_FIELDS = {
    "execution_mode": "STANDARD",
    "deep_executed": False,
    "plan_applied": False,
}

_INITIAL_META_KEYS = (
    "rerank_enabled",
    "rerank_applied",
    "rerank_model",
    "rerank_error",
    "rerank_strategy",
    "rerank_contract_version",
    "rerank_input_count",
    "rerank_output_count",
    "rerank_input_cap",
    "rerank_input_device_tier",
    "rerank_cache_enabled",
    "rerank_cache_hit",
    "retrieval_mode",
    "candidate_strategy",
    "candidate_strategy_family",
    "candidate_strategy_version",
    "candidate_strategy_fallback_from",
    "candidate_k",
    "leaf_retrieve_level",
    "auto_merge_enabled",
    "auto_merge_applied",
    "auto_merge_threshold",
    "auto_merge_replaced_chunks",
    "auto_merge_steps",
    "structure_rerank_enabled",
    "structure_rerank_applied",
    "structure_rerank_root_weight",
    "same_root_cap",
    "dominant_root_id",
    "dominant_root_share",
    "dominant_root_support",
    "confidence_gate_enabled",
    "fallback_required",
    "confidence_reasons",
    "top_margin",
    "top_score",
    "anchor_match",
    "query_anchors",
    "candidates_before_rerank",
    "candidates_after_rerank",
    "candidates_after_structure_rerank",
)


def normalize_trace_text(value: object) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def trace_text_hash(value: object, length: int = 16) -> str:
    text = normalize_trace_text(value)
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def candidate_identity(doc: dict[str, Any]) -> str:
    for key in ("chunk_id", "canonical_chunk_id", "doc_id"):
        value = doc.get(key)
        if value:
            return str(value)
    payload = "|".join(
        [
            str(doc.get("filename") or ""),
            str(doc.get("page_number") or doc.get("page_start") or ""),
            trace_text_hash(doc.get("retrieval_text") or doc.get("text") or ""),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def rerank_input_hash(doc: dict[str, Any]) -> str:
    text = doc.get("rerank_pair_text") or doc.get("retrieval_text") or doc.get("text") or ""
    return trace_text_hash(text)


def golden_trace_signature(query_id: str, variant: str, trace: dict[str, Any]) -> dict[str, Any]:
    before = list(trace.get("candidates_before_rerank") or [])
    after_rerank = list(trace.get("candidates_after_rerank") or [])
    final = list(trace.get("candidates_after_structure_rerank") or trace.get("retrieved_chunks") or [])
    return {
        "signature_version": TRACE_SIGNATURE_VERSION,
        "variant": variant,
        "query_id": query_id,
        "retrieval_mode": trace.get("retrieval_mode"),
        "fallback_required": trace.get("fallback_required"),
        "candidate_ids_before_rerank": [candidate_identity(doc) for doc in before],
        "rerank_input_hashes": [rerank_input_hash(doc) for doc in before],
        "candidate_ids_after_rerank": [candidate_identity(doc) for doc in after_rerank],
        "final_top5_chunk_ids": [candidate_identity(doc) for doc in final[:5]],
        "final_top5_file_pages": [
            {
                "filename": doc.get("filename"),
                "page_number": doc.get("page_number") or doc.get("page_start"),
            }
            for doc in final[:5]
        ],
        "candidate_count_before_rerank": trace.get("candidate_count_before_rerank"),
        "rerank_input_count": trace.get("rerank_input_count"),
    }


def append_stage_error(
    trace: MutableMapping[str, Any],
    stage: str,
    error: str,
    fallback_to: str | None = None,
    *,
    error_code: str | None = None,
    severity: str = "warning",
    recoverable: bool = True,
    user_visible: bool = False,
) -> StageErrorDict:
    item = StageError(
        stage=stage,
        error=error,
        fallback_to=fallback_to,
        error_code=error_code,
        severity=severity,
        recoverable=recoverable,
        user_visible=user_visible,
    ).as_dict()
    errors = list(trace.get("stage_errors") or [])
    errors.append(item)
    trace["stage_errors"] = errors
    return item


def mark_context_delivery(
    trace: Mapping[str, Any] | None,
    *,
    delivery_mode: str,
    context: str,
    docs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if trace is None:
        return None
    payload = dict(trace)
    payload["context_delivery_mode"] = delivery_mode
    payload["context_format_version"] = RAG_CONTEXT_FORMAT_VERSION
    payload["context_chars"] = len(context or "")
    payload["retrieved_chunk_count"] = len(docs or [])
    payload["final_context_chunk_count"] = len(docs or [])
    return payload


def build_retrieval_meta(
    base: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> RetrievalMeta:
    payload: dict[str, Any] = {
        **STANDARD_EXECUTION_FIELDS,
        "timings": {},
        "stage_errors": [],
    }
    if base:
        payload.update(dict(base))
    payload.update(overrides)
    payload["timings"] = dict(payload.get("timings") or {})
    payload["stage_errors"] = list(payload.get("stage_errors") or [])
    payload.setdefault("execution_mode", "STANDARD")
    payload.setdefault("deep_executed", False)
    payload.setdefault("plan_applied", False)
    return payload  # type: ignore[return-value]


def build_initial_rag_trace(
    *,
    query: str,
    docs: list[dict[str, Any]],
    context: str,
    retrieve_meta: Mapping[str, Any],
    context_files: list[str] | None = None,
    attached_docs: list[dict[str, Any]] | None = None,
    attached_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        **STANDARD_EXECUTION_FIELDS,
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "expanded_query": query,
        "retrieved_chunks": docs,
        "initial_retrieved_chunks": docs,
        "attached_context_chunks": list(attached_docs or []),
        "context_files": list(context_files or []),
        "retrieval_stage": "initial",
        "attached_context_count": int((attached_meta or {}).get("attached_context_count", 0) or 0),
        "timings": dict(retrieve_meta.get("timings") or {}),
        "stage_errors": list(retrieve_meta.get("stage_errors") or []),
        "context_chars": len(context),
        "retrieved_chunk_count": len(docs),
        "final_context_chunk_count": len(docs),
    }
    for key in _INITIAL_META_KEYS:
        trace[key] = retrieve_meta.get(key)
    return trace


def merge_expanded_rag_trace(
    trace: MutableMapping[str, Any],
    updates: Mapping[str, Any],
    *,
    timings: Mapping[str, float] | None = None,
    stage_errors: list[Mapping[str, Any]] | None = None,
) -> MutableMapping[str, Any]:
    trace.setdefault("execution_mode", "STANDARD")
    trace.setdefault("deep_executed", False)
    trace.setdefault("plan_applied", False)
    if timings:
        merged_timings = dict(trace.get("timings") or {})
        merged_timings.update(dict(timings))
        trace["timings"] = merged_timings
    if stage_errors:
        merged_errors = list(trace.get("stage_errors") or [])
        merged_errors.extend(dict(error) for error in stage_errors)
        trace["stage_errors"] = merged_errors
    trace.update(dict(updates))
    return trace
