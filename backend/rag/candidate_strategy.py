from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


PIPELINE_STAGE_MODEL = "rag-l0-l3-v1"
CANDIDATE_STRATEGY_VERSION = "candidate-strategy-v2"
SHARED_RERANK_CONTRACT_VERSION = "shared-rerank-v2"
SHARED_POSTPROCESS_CONTRACT_VERSION = "shared-postprocess-v1"


class CandidateStrategyId(str, Enum):
    STANDARD = "standard"
    LAYERED = "layered"
    DENSE_FALLBACK = "dense_fallback"


class CandidateStrategyDetail(str, Enum):
    GLOBAL_HYBRID = "global_hybrid"
    SCOPED_HYBRID = "scoped_hybrid"
    LAYERED_SPLIT = "layered_split"
    DENSE_FALLBACK = "dense_fallback"


class RerankExecutionMode(str, Enum):
    EXECUTED = "executed"
    DISABLED = "disabled"
    SKIPPED_CANDIDATE_ONLY = "skipped_candidate_only"
    FAILED_BEFORE_RERANK = "failed_before_rerank"
    FAILED_WITH_RANKED_CANDIDATES = "failed_with_ranked_candidates"


@dataclass(frozen=True)
class CandidateStrategyConfig:
    strategy: CandidateStrategyId = CandidateStrategyId.STANDARD
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateStrategyResult:
    requested: CandidateStrategyId
    effective: CandidateStrategyId
    detail: CandidateStrategyDetail
    fallback_from: CandidateStrategyId | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)


def normalize_candidate_strategy(value: str | None) -> CandidateStrategyConfig:
    raw = (value or "").strip().lower()
    if not raw:
        return CandidateStrategyConfig()
    if raw == CandidateStrategyId.STANDARD.value:
        return CandidateStrategyConfig(strategy=CandidateStrategyId.STANDARD)
    if raw == CandidateStrategyId.LAYERED.value:
        return CandidateStrategyConfig(strategy=CandidateStrategyId.LAYERED)
    return CandidateStrategyConfig(
        strategy=CandidateStrategyId.STANDARD,
        warnings=(f"invalid RAG_CANDIDATE_STRATEGY={value!r}; falling back to standard",),
    )


def candidate_strategy_trace(
    *,
    requested: CandidateStrategyId,
    effective: CandidateStrategyId,
    detail: CandidateStrategyDetail,
    fallback_from: CandidateStrategyId | None = None,
    rerank_execution_mode: RerankExecutionMode | None = None,
    warnings: tuple[str, ...] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pipeline_stage_model": PIPELINE_STAGE_MODEL,
        "candidate_strategy_requested": requested.value,
        "candidate_strategy_effective": effective.value,
        "candidate_strategy_version": CANDIDATE_STRATEGY_VERSION,
        "candidate_strategy_detail": detail.value,
        "rerank_contract": "shared_rerank",
        "rerank_contract_version": SHARED_RERANK_CONTRACT_VERSION,
        "postprocess_contract": "shared_retrieval_postprocess",
        "postprocess_contract_version": SHARED_POSTPROCESS_CONTRACT_VERSION,
    }
    if fallback_from:
        payload["candidate_strategy_fallback_from"] = fallback_from.value
    if rerank_execution_mode:
        payload["rerank_execution_mode"] = rerank_execution_mode.value
    if warnings:
        payload["candidate_strategy_warnings"] = list(warnings)
    return payload
