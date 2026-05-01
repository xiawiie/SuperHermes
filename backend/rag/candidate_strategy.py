from __future__ import annotations

from enum import Enum
from typing import Any


CANDIDATE_STRATEGY_VERSION = "candidate-strategy-v1"
SHARED_RERANK_CONTRACT_VERSION = "shared-rerank-v1"


class CandidateStrategy(str, Enum):
    GLOBAL_HYBRID = "global_hybrid"
    SCOPED_HYBRID = "scoped_hybrid"
    LAYERED_SPLIT = "layered_split"
    DENSE_FALLBACK = "dense_fallback"


class CandidateStrategyFamily(str, Enum):
    STANDARD = "standard"
    LAYERED = "layered"


class RerankStrategy(str, Enum):
    SHARED_PIPELINE = "shared_pipeline"


def candidate_strategy_trace(
    strategy: CandidateStrategy,
    family: CandidateStrategyFamily,
    *,
    fallback_from: CandidateStrategy | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_strategy": strategy.value,
        "candidate_strategy_family": family.value,
        "candidate_strategy_version": CANDIDATE_STRATEGY_VERSION,
        "rerank_strategy": RerankStrategy.SHARED_PIPELINE.value,
        "rerank_contract_version": SHARED_RERANK_CONTRACT_VERSION,
    }
    if fallback_from:
        payload["candidate_strategy_fallback_from"] = fallback_from.value
    return payload
