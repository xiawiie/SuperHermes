from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.rag.citations import build_citation_refs, format_evidence_with_refs, verify_citations
from backend.rag.runtime_config import RagRuntimeConfig, load_runtime_config


RetrieveFn = Callable[[str, list[str] | None], dict[str, Any]]
SynthesizeFn = Callable[[str, str, list[dict[str, Any]]], str]


@dataclass(frozen=True)
class DeepModeRequest:
    question: str
    context_files: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    active: bool = False


@dataclass(frozen=True)
class SubqueryEvidence:
    subquery: str
    docs: list[dict[str, Any]]
    context: str
    rag_trace: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "subquery": self.subquery,
            "docs": self.docs,
            "context": self.context,
            "rag_trace": self.rag_trace,
        }


@dataclass(frozen=True)
class DeepModeResult:
    final_answer: str
    citations: list[dict[str, Any]]
    evidence: list[SubqueryEvidence]
    evidence_coverage: float
    deep_executed: bool
    fallback_to_standard: bool
    fallback_reason: str | None
    rag_trace: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "citations": self.citations,
            "evidence": [item.as_dict() for item in self.evidence],
            "evidence_by_subquery": _evidence_by_subquery(self.evidence),
            "coverage_by_subquery": _coverage_by_subquery(self.evidence),
            "retrieval_trace_by_subquery": _retrieval_trace_by_subquery(self.evidence),
            "answer_mode": self.rag_trace.get("answer_mode"),
            "evidence_coverage": self.evidence_coverage,
            "deep_executed": self.deep_executed,
            "fallback_to_standard": self.fallback_to_standard,
            "fallback_reason": self.fallback_reason,
            "rag_trace": self.rag_trace,
        }


def plan_subqueries(question: str, explicit_subqueries: Sequence[str] | None = None) -> list[str]:
    subqueries = [item.strip() for item in explicit_subqueries or [] if item and item.strip()]
    return subqueries or [question]


def deep_mode_policy(config: RagRuntimeConfig | None = None) -> str:
    config = config or load_runtime_config()
    if config.deep_active_enabled:
        return "active"
    if config.deep_shadow_enabled:
        return "shadow"
    return "disabled"


def _default_retrieve(query: str, context_files: list[str] | None = None) -> dict[str, Any]:
    from backend.rag.pipeline import run_rag_graph

    return run_rag_graph(query, context_files=context_files)


def _collect_evidence(
    request: DeepModeRequest,
    *,
    retrieve: RetrieveFn,
) -> list[SubqueryEvidence]:
    evidence = []
    for subquery in plan_subqueries(request.question, request.subqueries):
        result = retrieve(subquery, request.context_files)
        docs = result.get("docs", []) if isinstance(result, dict) else []
        context = result.get("context", "") if isinstance(result, dict) else ""
        rag_trace = result.get("rag_trace", {}) if isinstance(result, dict) else {}
        evidence.append(
            SubqueryEvidence(
                subquery=subquery,
                docs=docs,
                context=context,
                rag_trace=rag_trace,
            )
        )
    return evidence


def _evidence_coverage(evidence: Sequence[SubqueryEvidence]) -> float:
    if not evidence:
        return 0.0
    covered = sum(1 for item in evidence if item.docs)
    return covered / len(evidence)


def _flatten_evidence_docs(evidence: Sequence[SubqueryEvidence]) -> list[dict[str, Any]]:
    docs = []
    seen = set()
    for item in evidence:
        for doc in item.docs:
            key = doc.get("chunk_id") or (doc.get("filename"), doc.get("page_number"), doc.get("text"))
            if key in seen:
                continue
            seen.add(key)
            docs.append(doc)
    return docs


def _evidence_by_subquery(evidence: Sequence[SubqueryEvidence]) -> dict[str, list[dict[str, Any]]]:
    return {item.subquery: item.docs for item in evidence}


def _coverage_by_subquery(evidence: Sequence[SubqueryEvidence]) -> dict[str, bool]:
    return {item.subquery: bool(item.docs) for item in evidence}


def _retrieval_trace_by_subquery(evidence: Sequence[SubqueryEvidence]) -> dict[str, dict[str, Any]]:
    return {item.subquery: item.rag_trace for item in evidence}


def _deep_trace(
    *,
    mode: str,
    evidence: Sequence[SubqueryEvidence],
    coverage: float,
    fallback_to_standard: bool,
    fallback_reason: str | None,
    citation_verifier: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "execution_mode": "DEEP",
        "deep_mode": mode,
        "answer_mode": mode,
        "deep_executed": True,
        "plan_applied": bool(evidence),
        "subqueries": [item.subquery for item in evidence],
        "coverage_by_subquery": _coverage_by_subquery(evidence),
        "retrieval_trace_by_subquery": _retrieval_trace_by_subquery(evidence),
        "evidence_coverage": coverage,
        "fallback_to_standard": fallback_to_standard,
        "fallback_reason": fallback_reason,
        "citation_verifier": citation_verifier,
    }


def run_deep_mode(
    request: DeepModeRequest,
    *,
    retrieve: RetrieveFn | None = None,
    synthesize: SynthesizeFn | None = None,
    config: RagRuntimeConfig | None = None,
) -> DeepModeResult:
    config = config or load_runtime_config()
    configured_mode = deep_mode_policy(config)
    mode = "active" if request.active and config.deep_active_enabled else configured_mode
    if mode == "disabled":
        return DeepModeResult(
            final_answer="",
            citations=[],
            evidence=[],
            evidence_coverage=0.0,
            deep_executed=False,
            fallback_to_standard=True,
            fallback_reason="deep_mode_disabled",
            rag_trace={
                "execution_mode": "STANDARD",
                "deep_executed": False,
                "deep_mode": "disabled",
                "answer_mode": "disabled",
                "fallback_to_standard": True,
                "fallback_reason": "deep_mode_disabled",
            },
        )

    evidence = _collect_evidence(request, retrieve=retrieve or _default_retrieve)
    coverage = _evidence_coverage(evidence)
    docs = _flatten_evidence_docs(evidence)
    refs = build_citation_refs(docs)
    citations = [ref.as_dict() for ref in refs]

    if coverage < config.deep_min_coverage:
        reason = "insufficient_evidence_coverage"
        return DeepModeResult(
            final_answer="",
            citations=citations,
            evidence=evidence,
            evidence_coverage=coverage,
            deep_executed=True,
            fallback_to_standard=True,
            fallback_reason=reason,
            rag_trace=_deep_trace(
                mode=mode,
                evidence=evidence,
                coverage=coverage,
                fallback_to_standard=True,
                fallback_reason=reason,
            ),
        )

    if mode == "shadow":
        return DeepModeResult(
            final_answer="",
            citations=citations,
            evidence=evidence,
            evidence_coverage=coverage,
            deep_executed=True,
            fallback_to_standard=True,
            fallback_reason="shadow_mode",
            rag_trace=_deep_trace(
                mode=mode,
                evidence=evidence,
                coverage=coverage,
                fallback_to_standard=True,
                fallback_reason="shadow_mode",
            ),
        )

    if not config.citation_verify_enabled:
        reason = "citation_verifier_disabled"
        return DeepModeResult(
            final_answer="",
            citations=citations,
            evidence=evidence,
            evidence_coverage=coverage,
            deep_executed=True,
            fallback_to_standard=True,
            fallback_reason=reason,
            rag_trace=_deep_trace(
                mode=mode,
                evidence=evidence,
                coverage=coverage,
                fallback_to_standard=True,
                fallback_reason=reason,
            ),
        )

    if synthesize is None:
        reason = "synthesizer_unavailable"
        return DeepModeResult(
            final_answer="",
            citations=citations,
            evidence=evidence,
            evidence_coverage=coverage,
            deep_executed=True,
            fallback_to_standard=True,
            fallback_reason=reason,
            rag_trace=_deep_trace(
                mode=mode,
                evidence=evidence,
                coverage=coverage,
                fallback_to_standard=True,
                fallback_reason=reason,
            ),
        )

    evidence_text = format_evidence_with_refs(docs)
    answer = synthesize(request.question, evidence_text, [ref.as_dict() for ref in refs])
    verifier = verify_citations(answer, refs, require_citations=True).as_dict()
    if not verifier["valid"]:
        reason = "citation_verification_failed"
        return DeepModeResult(
            final_answer="",
            citations=citations,
            evidence=evidence,
            evidence_coverage=coverage,
            deep_executed=True,
            fallback_to_standard=True,
            fallback_reason=reason,
            rag_trace=_deep_trace(
                mode=mode,
                evidence=evidence,
                coverage=coverage,
                fallback_to_standard=True,
                fallback_reason=reason,
                citation_verifier=verifier,
            ),
        )

    return DeepModeResult(
        final_answer=answer,
        citations=citations,
        evidence=evidence,
        evidence_coverage=coverage,
        deep_executed=True,
        fallback_to_standard=False,
        fallback_reason=None,
        rag_trace=_deep_trace(
            mode=mode,
            evidence=evidence,
            coverage=coverage,
            fallback_to_standard=False,
            fallback_reason=None,
            citation_verifier=verifier,
        ),
    )
