from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


RetrievalStage = Literal[
    "query_plan",
    "global_hybrid",
    "scoped_hybrid",
    "dense_fallback",
    "rerank",
    "structure_rerank",
    "confidence_gate",
]


class RetrievedDocument(TypedDict, total=False):
    chunk_id: str
    parent_chunk_id: str
    root_chunk_id: str
    canonical_chunk_id: str
    canonical_root_id: str
    filename: str
    page_number: int
    page_start: int
    page_end: int
    section_title: str
    section_path: str
    anchor_id: str
    text: str
    retrieval_text: str
    score: float
    rerank_score: float
    raw_rerank_score: float
    fusion_score: float
    final_score: float
    index_profile: str


@dataclass(frozen=True)
class StageError:
    stage: str
    error: str
    fallback_to: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {"stage": self.stage, "error": self.error}
        if self.fallback_to:
            payload["fallback_to"] = self.fallback_to
        return payload


@dataclass
class RetrievalTrace:
    retrieval_mode: str = ""
    candidate_k: int = 0
    candidate_count_before_rerank: int = 0
    candidate_count_after_rerank: int = 0
    candidate_count_after_structure_rerank: int = 0
    fallback_required: bool | None = None
    timings: dict[str, float] = field(default_factory=dict)
    stage_errors: list[dict[str, str]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "retrieval_mode": self.retrieval_mode,
            "candidate_k": self.candidate_k,
            "candidate_count_before_rerank": self.candidate_count_before_rerank,
            "candidate_count_after_rerank": self.candidate_count_after_rerank,
            "candidate_count_after_structure_rerank": self.candidate_count_after_structure_rerank,
            "fallback_required": self.fallback_required,
            "timings": dict(self.timings),
            "stage_errors": list(self.stage_errors),
        }
        payload.update(self.extra)
        return payload
