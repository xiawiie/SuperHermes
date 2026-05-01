from __future__ import annotations

from collections.abc import Sequence
from typing import Any


CHUNK_SEPARATOR = "\n\n---\n\n"
NO_RELEVANT_DOCUMENTS_MESSAGE = "No relevant documents found in the knowledge base."
RETRIEVED_CHUNKS_HEADER = "Retrieved Chunks:"


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


def format_rag_tool_response(docs: Sequence[dict[str, Any]], *, context: str | None = None) -> str:
    if not docs:
        return NO_RELEVANT_DOCUMENTS_MESSAGE
    body = context or format_rag_documents(docs)
    if not body:
        return NO_RELEVANT_DOCUMENTS_MESSAGE
    return f"{RETRIEVED_CHUNKS_HEADER}\n{body}"
