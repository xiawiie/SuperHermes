from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from backend.rag.candidate_strategy import CandidateStrategyConfig, normalize_candidate_strategy
from backend.rag.profile_naming import resolve_runtime_dtype
from backend.rag.profiles import normalize_index_profile


EnvMapping = Mapping[str, str | None]


def _value(env: EnvMapping, name: str, default: str | None = None) -> str | None:
    value = env.get(name)
    return default if value is None else value


def _str(env: EnvMapping, name: str, default: str = "") -> str:
    value = _value(env, name, default)
    return str(value or "")


def _lower(env: EnvMapping, name: str, default: str = "") -> str:
    return _str(env, name, default).lower()


def _int(env: EnvMapping, name: str, default: int = 0) -> int:
    raw = _value(env, name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(str(raw).strip())


def _float(env: EnvMapping, name: str, default: float = 0.0) -> float:
    raw = _value(env, name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(str(raw).strip())


def _bool(env: EnvMapping, name: str, default: bool = False) -> bool:
    raw = _value(env, name)
    if raw is None:
        return default
    return str(raw).strip().lower() == "true"


def _is_reserved_flag(name: str) -> bool:
    return name == "RAG_MODE" or name.startswith(("RAG_MODE_", "RAG_FAST_", "RAG_DEEP_"))


def _bounded_float(env: EnvMapping, name: str, default: float, *, low: float, high: float) -> float:
    return min(high, max(low, _float(env, name, default)))


@dataclass(frozen=True)
class RagRuntimeConfig:
    rerank_model: str | None = None
    rerank_provider: str = "local"
    rerank_device: str = "auto"
    rerank_torch_dtype: str = "float16"
    rerank_input_k_cpu: int = 0
    auto_merge_enabled: bool = True
    auto_merge_threshold: int = 2
    leaf_retrieve_level: int = 3
    structure_rerank_root_weight: float = 0.3
    same_root_cap: int = 2
    structure_rerank_enabled: bool = True
    confidence_gate_enabled: bool = False
    low_conf_top_margin: float = 0.05
    low_conf_root_share: float = 0.45
    low_conf_top_score: float = 0.20
    enable_anchor_gate: bool = True
    query_plan_enabled: bool = False
    rag_candidate_k: int = 0
    rerank_top_n: int = 0
    rerank_input_k_gpu: int = 0
    rerank_cache_enabled: bool = True
    rerank_cache_ttl_seconds: int = 300
    milvus_search_ef: int = 64
    milvus_sparse_drop_ratio: float = 0.2
    milvus_rrf_k: int = 60
    doc_scope_global_reserve_weight: float = 0.2
    doc_scope_filename_boost_weight: float = 0.15
    rerank_pair_enrichment_enabled: bool = False
    heading_lexical_enabled: bool = False
    heading_lexical_weight: float = 0.20
    rag_index_profile: str = ""
    rerank_score_fusion_enabled: bool = False
    rerank_fusion_rerank_weight: float = 0.65
    rerank_fusion_rrf_weight: float = 0.20
    rerank_fusion_scope_weight: float = 0.10
    rerank_fusion_metadata_weight: float = 0.05
    citation_verify_enabled: bool = False
    unified_execution_enabled: bool = False
    deep_shadow_enabled: bool = False
    deep_active_enabled: bool = False
    deep_min_coverage: float = 0.75
    candidate_strategy: CandidateStrategyConfig = field(default_factory=CandidateStrategyConfig)
    execution_mode: str = "STANDARD"
    deep_executed: bool = False
    plan_applied: bool = False
    reserved_flags: dict[str, str] = field(default_factory=dict)


def load_candidate_strategy_config(env: EnvMapping | None = None) -> CandidateStrategyConfig:
    env = os.environ if env is None else env
    return normalize_candidate_strategy(_value(env, "RAG_CANDIDATE_STRATEGY"))


def load_runtime_config(env: EnvMapping | None = None) -> RagRuntimeConfig:
    env = os.environ if env is None else env
    reserved_flags = {
        name: str(value)
        for name, value in env.items()
        if value is not None and _is_reserved_flag(name)
    }
    rerank_model = _value(env, "RERANK_MODEL") or _value(env, "RERANK_AVAILABLE_MODEL")
    return RagRuntimeConfig(
        rerank_model=str(rerank_model) if rerank_model else None,
        rerank_provider=_lower(env, "RERANK_PROVIDER", "local"),
        rerank_device=_str(env, "RERANK_DEVICE", _str(env, "RAG_A", "auto")),
        rerank_torch_dtype=resolve_runtime_dtype(_value(env, "RAG_DTYPE"), _value(env, "RERANK_TORCH_DTYPE")),
        rerank_input_k_cpu=_int(env, "RERANK_INPUT_K_CPU", 0),
        auto_merge_enabled=_str(env, "AUTO_MERGE_ENABLED", "true").lower() != "false",
        auto_merge_threshold=_int(env, "AUTO_MERGE_THRESHOLD", 2),
        leaf_retrieve_level=_int(env, "LEAF_RETRIEVE_LEVEL", 3),
        structure_rerank_root_weight=_float(env, "STRUCTURE_RERANK_ROOT_WEIGHT", 0.3),
        same_root_cap=_int(env, "SAME_ROOT_CAP", 2),
        structure_rerank_enabled=_str(env, "STRUCTURE_RERANK_ENABLED", "true").lower() != "false",
        confidence_gate_enabled=_bool(env, "CONFIDENCE_GATE_ENABLED", False),
        low_conf_top_margin=_float(env, "LOW_CONF_TOP_MARGIN", 0.05),
        low_conf_root_share=_float(env, "LOW_CONF_ROOT_SHARE", 0.45),
        low_conf_top_score=_float(env, "LOW_CONF_TOP_SCORE", 0.20),
        enable_anchor_gate=_str(env, "ENABLE_ANCHOR_GATE", "true").lower() != "false",
        query_plan_enabled=_bool(env, "QUERY_PLAN_ENABLED", False),
        rag_candidate_k=_int(env, "RAG_CANDIDATE_K", 0),
        rerank_top_n=_int(env, "RERANK_TOP_N", 0),
        rerank_input_k_gpu=_int(env, "RERANK_INPUT_K_GPU", 0),
        rerank_cache_enabled=_bool(env, "RERANK_CACHE_ENABLED", True),
        rerank_cache_ttl_seconds=_int(env, "RERANK_CACHE_TTL_SECONDS", 300),
        milvus_search_ef=_int(env, "MILVUS_SEARCH_EF", 64),
        milvus_sparse_drop_ratio=_float(env, "MILVUS_SPARSE_DROP_RATIO", 0.2),
        milvus_rrf_k=_int(env, "MILVUS_RRF_K", 60),
        doc_scope_global_reserve_weight=_bounded_float(
            env,
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT",
            0.2,
            low=0.0,
            high=1.0,
        ),
        doc_scope_filename_boost_weight=_float(env, "DOC_SCOPE_FILENAME_BOOST_WEIGHT", 0.15),
        rerank_pair_enrichment_enabled=_bool(env, "RERANK_PAIR_ENRICHMENT_ENABLED", False),
        heading_lexical_enabled=_bool(env, "HEADING_LEXICAL_ENABLED", False),
        heading_lexical_weight=_bounded_float(env, "HEADING_LEXICAL_WEIGHT", 0.20, low=0.0, high=1.0),
        rag_index_profile=normalize_index_profile(_value(env, "RAG_INDEX_PROFILE")),
        rerank_score_fusion_enabled=_bool(env, "RERANK_SCORE_FUSION_ENABLED", False),
        rerank_fusion_rerank_weight=_float(env, "RERANK_FUSION_RERANK_WEIGHT", 0.65),
        rerank_fusion_rrf_weight=_float(env, "RERANK_FUSION_RRF_WEIGHT", 0.20),
        rerank_fusion_scope_weight=_float(env, "RERANK_FUSION_SCOPE_WEIGHT", 0.10),
        rerank_fusion_metadata_weight=_float(env, "RERANK_FUSION_METADATA_WEIGHT", 0.05),
        citation_verify_enabled=_bool(env, "RAG_CITATION_VERIFY_ENABLED", False),
        unified_execution_enabled=_bool(env, "RAG_UNIFIED_EXECUTION_ENABLED", False),
        deep_shadow_enabled=_bool(env, "RAG_DEEP_SHADOW", False),
        deep_active_enabled=_bool(env, "RAG_DEEP_ACTIVE", False),
        deep_min_coverage=_bounded_float(env, "RAG_DEEP_MIN_COVERAGE", 0.75, low=0.0, high=1.0),
        candidate_strategy=load_candidate_strategy_config(env),
        reserved_flags=reserved_flags,
    )
