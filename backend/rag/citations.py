from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from backend.rag.formatting import CHUNK_SEPARATOR
from backend.rag.runtime_config import load_runtime_config


_CITATION_RE = re.compile(
    r"\[(C\d+)"
    r"(?:\|file=([^\]|]+))?"
    r"(?:\|page=([^\]]+))?"
    r"\]"
)


@dataclass(frozen=True)
class CitationRef:
    ref_id: str
    chunk_id: str | None
    filename: str
    page_number: int | str | None
    section_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "chunk_id": self.chunk_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "section_path": self.section_path,
        }


@dataclass(frozen=True)
class CitationVerifierResult:
    valid: bool
    cited_refs: list[str]
    unknown_refs: list[str]
    missing_required_refs: list[str]
    metadata_mismatches: list[dict[str, Any]]
    citation_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "cited_refs": self.cited_refs,
            "unknown_refs": self.unknown_refs,
            "missing_required_refs": self.missing_required_refs,
            "metadata_mismatches": self.metadata_mismatches,
            "citation_error": self.citation_error,
        }


def _doc_page(doc: Mapping[str, Any]) -> int | str | None:
    return doc.get("page_number") or doc.get("page_start") or doc.get("page")


def build_citation_refs(docs: Sequence[Mapping[str, Any]]) -> list[CitationRef]:
    refs = []
    for idx, doc in enumerate(docs or [], 1):
        refs.append(
            CitationRef(
                ref_id=f"C{idx}",
                chunk_id=str(doc.get("chunk_id")) if doc.get("chunk_id") else None,
                filename=str(doc.get("filename") or "Unknown"),
                page_number=_doc_page(doc),
                section_path=str(doc.get("section_path")) if doc.get("section_path") else None,
            )
        )
    return refs


def format_evidence_with_refs(docs: Sequence[Mapping[str, Any]]) -> str:
    chunks = []
    for ref, doc in zip(build_citation_refs(docs), docs or []):
        page = ref.page_number if ref.page_number is not None else "N/A"
        text = str(doc.get("text") or "")
        chunks.append(f"[{ref.ref_id}] {ref.filename} (Page {page}):\n{text}")
    return CHUNK_SEPARATOR.join(chunks)


def _normalize_page(value: object) -> str:
    return str(value or "").strip().lower()


def verify_citations(
    answer: str,
    refs: Sequence[CitationRef],
    *,
    require_citations: bool = False,
) -> CitationVerifierResult:
    ref_by_id = {ref.ref_id: ref for ref in refs}
    cited_refs = []
    unknown_refs = []
    metadata_mismatches: list[dict[str, Any]] = []

    for match in _CITATION_RE.finditer(answer or ""):
        ref_id, cited_file, cited_page = match.groups()
        if ref_id not in cited_refs:
            cited_refs.append(ref_id)
        ref = ref_by_id.get(ref_id)
        if ref is None:
            if ref_id not in unknown_refs:
                unknown_refs.append(ref_id)
            continue
        if cited_file and cited_file.strip() != ref.filename:
            metadata_mismatches.append(
                {
                    "ref_id": ref_id,
                    "field": "filename",
                    "expected": ref.filename,
                    "actual": cited_file.strip(),
                }
            )
        if cited_page and _normalize_page(cited_page) != _normalize_page(ref.page_number):
            metadata_mismatches.append(
                {
                    "ref_id": ref_id,
                    "field": "page_number",
                    "expected": ref.page_number,
                    "actual": cited_page.strip(),
                }
            )

    missing_required_refs = ["any"] if require_citations and refs and not cited_refs else []
    valid = not unknown_refs and not metadata_mismatches and not missing_required_refs
    return CitationVerifierResult(
        valid=valid,
        cited_refs=cited_refs,
        unknown_refs=unknown_refs,
        missing_required_refs=missing_required_refs,
        metadata_mismatches=metadata_mismatches,
    )


def maybe_verify_citation_trace(
    answer: str,
    trace: Mapping[str, Any] | None,
    *,
    enabled: bool | None = None,
    require_citations: bool = False,
) -> dict[str, Any] | None:
    if trace is None:
        return None
    if enabled is None:
        enabled = load_runtime_config().citation_verify_enabled
    if not enabled:
        return dict(trace)

    payload = dict(trace)
    docs = payload.get("retrieved_chunks") or payload.get("initial_retrieved_chunks") or []
    refs = build_citation_refs(docs if isinstance(docs, list) else [])
    result = verify_citations(answer, refs, require_citations=require_citations)
    payload["citation_verifier"] = {
        **result.as_dict(),
        "refs": [ref.as_dict() for ref in refs],
        "mode": "deterministic_ref_check",
    }
    return payload
