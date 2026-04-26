from __future__ import annotations

import hashlib
import re
from typing import Any


TRACE_SIGNATURE_VERSION = "rag-trace-signature-v1"


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
