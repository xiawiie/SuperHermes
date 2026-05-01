from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CHUNK_SEPARATOR = "\n\n---\n\n"
NO_RELEVANT_DOCUMENTS_MESSAGE = "No relevant documents found in the knowledge base."
RETRIEVED_CHUNKS_HEADER = "Retrieved Chunks:"
TOOL_META_HEADER = "Retrieval Metadata:"
TOOL_META_KEYS = (
    "candidate_strategy_requested",
    "candidate_strategy_effective",
    "candidate_strategy_detail",
    "rerank_contract_version",
    "postprocess_contract_version",
    "rerank_execution_mode",
    "context_delivery_mode",
)


def format_rag_documents(docs: Sequence[dict[str, Any]]) -> str:
    if not docs:
        return ""
    chunks = []
    for index, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        page = doc.get("page_number", "N/A")
        text = doc.get("text", "")
        chunks.append(f"[{index}] {source} (Page {page}):\n{text}")
    return CHUNK_SEPARATOR.join(chunks)


def _format_tool_retrieval_meta(retrieval_meta: Mapping[str, Any] | None) -> str:
    if not retrieval_meta:
        return ""
    has_retrieval_contract = any(
        retrieval_meta.get(key) is not None for key in TOOL_META_KEYS if key != "context_delivery_mode"
    )
    if not has_retrieval_contract:
        return ""
    lines = []
    for key in TOOL_META_KEYS:
        value = retrieval_meta.get(key)
        if value is None:
            continue
        lines.append(f"{key}={value}")
    return "\n".join(lines)


def format_rag_tool_response(
    docs: Sequence[dict[str, Any]],
    *,
    context: str | None = None,
    retrieval_meta: Mapping[str, Any] | None = None,
) -> str:
    if not docs:
        return NO_RELEVANT_DOCUMENTS_MESSAGE
    body = context or format_rag_documents(docs)
    if not body:
        return NO_RELEVANT_DOCUMENTS_MESSAGE
    meta = _format_tool_retrieval_meta(retrieval_meta)
    if meta:
        return f"{TOOL_META_HEADER}\n{meta}\n\n{RETRIEVED_CHUNKS_HEADER}\n{body}"
    return f"{RETRIEVED_CHUNKS_HEADER}\n{body}"
