from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import json
import hashlib
import re
import time

from backend.infra.cache import cache
from backend.shared.json_utils import extract_json_object
from backend.infra.vector_store.milvus_client import MilvusManager
from backend.infra.embedding import embedding_service as _embedding_service
from backend.rag.query_plan import (
    QueryPlan,
    DOC_SCOPE_MATCH_BOOST,
    parse_query_plan,
    get_filename_registry,
    _normalize_filename as _qp_normalize_filename,
)
from backend.rag.confidence import (
    doc_matches_anchor as _confidence_doc_matches_anchor,
    evaluate_retrieval_confidence as _confidence_evaluate_retrieval_confidence,
    extract_query_anchors as _confidence_extract_query_anchors,
)
from backend.rag.context import (
    apply_structure_rerank as _context_apply_structure_rerank,
    auto_merge_documents as _context_auto_merge_documents,
    merge_to_parent_level as _context_merge_to_parent_level,
)
from backend.rag.rerank import (
    RerankRuntime,
    apply_rerank_score_fusion as _rerank_apply_score_fusion,
    build_enriched_pair as _rerank_build_enriched_pair,
    rerank_documents as _run_rerank_documents,
    rerank_pair_text as _rerank_pair_text_impl,
    rerank_rrf_score as _rerank_rrf_score_impl,
)
from backend.rag.candidate_strategy import (
    CandidateStrategy,
    CandidateStrategyFamily,
    candidate_strategy_trace,
)
from backend.rag.retrieval import (
    annotate_scope_scores as _retrieval_annotate_scope_scores,
    apply_filename_boost as _retrieval_apply_filename_boost,
    apply_heading_lexical_scoring as _retrieval_apply_heading_lexical_scoring,
    build_filename_filter as _retrieval_build_filename_filter,
    dedupe_docs as _retrieval_dedupe_docs,
    weighted_rrf_merge as _retrieval_weighted_rrf_merge,
)
from backend.rag.layered_rerank import (
    build_l1_candidates as _build_l1_candidates,
)
from backend.rag.runtime_config import LayeredRerankConfig, RagRuntimeConfig, load_runtime_config
from backend.rag.trace import build_retrieval_meta, candidate_identity, trace_text_hash
from backend.rag.types import StageError, StageErrorDict
from backend.rag.profile_naming import resolve_dtype as resolve_dtype
from backend.config import (
    ARK_API_KEY,
    BASE_URL,
    MODEL,
)


_RUNTIME_CONFIG = load_runtime_config()
_LAYERED_CONFIG = _RUNTIME_CONFIG.layered

RERANK_MODEL = _RUNTIME_CONFIG.rerank_model
RERANK_PROVIDER = _RUNTIME_CONFIG.rerank_provider
RERANK_DEVICE = _RUNTIME_CONFIG.rerank_device
RERANK_TORCH_DTYPE = _RUNTIME_CONFIG.rerank_torch_dtype
RERANK_INPUT_K_CPU = _RUNTIME_CONFIG.rerank_input_k_cpu
AUTO_MERGE_ENABLED = _RUNTIME_CONFIG.auto_merge_enabled
AUTO_MERGE_THRESHOLD = _RUNTIME_CONFIG.auto_merge_threshold
LEAF_RETRIEVE_LEVEL = _RUNTIME_CONFIG.leaf_retrieve_level
STRUCTURE_RERANK_ROOT_WEIGHT = _RUNTIME_CONFIG.structure_rerank_root_weight
SAME_ROOT_CAP = _RUNTIME_CONFIG.same_root_cap
STRUCTURE_RERANK_ENABLED = _RUNTIME_CONFIG.structure_rerank_enabled
CONFIDENCE_GATE_ENABLED = _RUNTIME_CONFIG.confidence_gate_enabled
LOW_CONF_TOP_MARGIN = _RUNTIME_CONFIG.low_conf_top_margin
LOW_CONF_ROOT_SHARE = _RUNTIME_CONFIG.low_conf_root_share
LOW_CONF_TOP_SCORE = _RUNTIME_CONFIG.low_conf_top_score
ENABLE_ANCHOR_GATE = _RUNTIME_CONFIG.enable_anchor_gate
QUERY_PLAN_ENABLED = _RUNTIME_CONFIG.query_plan_enabled
RAG_CANDIDATE_K = _RUNTIME_CONFIG.rag_candidate_k
RERANK_TOP_N = _RUNTIME_CONFIG.rerank_top_n
RERANK_INPUT_K_GPU = _RUNTIME_CONFIG.rerank_input_k_gpu
RERANK_CACHE_ENABLED = _RUNTIME_CONFIG.rerank_cache_enabled
RERANK_CACHE_TTL_SECONDS = _RUNTIME_CONFIG.rerank_cache_ttl_seconds
MILVUS_SEARCH_EF = _RUNTIME_CONFIG.milvus_search_ef
MILVUS_SPARSE_DROP_RATIO = _RUNTIME_CONFIG.milvus_sparse_drop_ratio
MILVUS_RRF_K = _RUNTIME_CONFIG.milvus_rrf_k
DOC_SCOPE_GLOBAL_RESERVE_WEIGHT = _RUNTIME_CONFIG.doc_scope_global_reserve_weight
DOC_SCOPE_FILENAME_BOOST_WEIGHT = _RUNTIME_CONFIG.doc_scope_filename_boost_weight
RERANK_PAIR_ENRICHMENT_ENABLED = _RUNTIME_CONFIG.rerank_pair_enrichment_enabled
HEADING_LEXICAL_ENABLED = _RUNTIME_CONFIG.heading_lexical_enabled
HEADING_LEXICAL_WEIGHT = _RUNTIME_CONFIG.heading_lexical_weight
RAG_INDEX_PROFILE = _RUNTIME_CONFIG.rag_index_profile
RERANK_SCORE_FUSION_ENABLED = _RUNTIME_CONFIG.rerank_score_fusion_enabled
RERANK_FUSION_RERANK_WEIGHT = _RUNTIME_CONFIG.rerank_fusion_rerank_weight
RERANK_FUSION_RRF_WEIGHT = _RUNTIME_CONFIG.rerank_fusion_rrf_weight
RERANK_FUSION_SCOPE_WEIGHT = _RUNTIME_CONFIG.rerank_fusion_scope_weight
RERANK_FUSION_METADATA_WEIGHT = _RUNTIME_CONFIG.rerank_fusion_metadata_weight

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = MilvusManager()
_parent_chunk_store = None

_stepback_model = None
_local_reranker = None
_ANCHOR_PATTERN = re.compile(
    r"(第[一二三四五六七八九十百千万零两0-9]+[编章节条部分款项]|"
    r"\d+(?:\.\d+){1,4}|"
    r"[一二三四五六七八九十]+、|"
    r"[（(][一二三四五六七八九十0-9A-Za-z]+[)）]|"
    r"附录[A-Za-z0-9一二三四五六七八九十]+|"
    r"附件[0-9一二三四五六七八九十]+)"
)


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = 5
    context_files: list[str] = field(default_factory=list)
    config: RagRuntimeConfig = field(default_factory=lambda: _RUNTIME_CONFIG)

    @property
    def layered_config(self) -> LayeredRerankConfig:
        return self.config.layered


@dataclass(frozen=True)
class RetrievalFilters:
    base_filter: str
    filename_filter: str
    effective_filter: str
    scoped_filter: str | None = None
    matched_files: list[tuple[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class QueryEmbeddings:
    dense: list[float] | None
    sparse: Any | None
    dense_error: str | None = None
    sparse_error: str | None = None

    @property
    def has_dense(self) -> bool:
        return self.dense is not None

    @property
    def has_sparse(self) -> bool:
        return self.sparse is not None


@dataclass
class CandidateRetrievalResult:
    candidates: list[dict]
    retrieval_mode: str
    trace_patch: dict[str, Any] = field(default_factory=dict)
    stage_errors: list[StageErrorDict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    hybrid_error: str | None = None


@dataclass
class PreparedRetrieval:
    query: str
    search_query: str
    top_k: int
    candidate_k: int
    context_files: list[str]
    query_plan: QueryPlan
    filters: RetrievalFilters
    candidates: list[dict]
    retrieval_mode: str
    trace_patch: dict[str, Any]
    timings: Dict[str, float]
    stage_errors: list[StageErrorDict]
    total_start: float
    hybrid_error: str | None = None
    dense_error: str | None = None

    @property
    def failed(self) -> bool:
        return self.retrieval_mode == "failed"


def _get_parent_chunk_store():
    global _parent_chunk_store
    if _parent_chunk_store is None:
        from backend.infra.vector_store.parent_chunk_store import ParentChunkStore

        _parent_chunk_store = ParentChunkStore()
    return _parent_chunk_store


def _doc_retrieval_text(doc: dict) -> str:
    return str(doc.get("retrieval_text") or doc.get("text") or "")


def _build_enriched_pair(doc: dict) -> str:
    return _rerank_build_enriched_pair(doc, doc_text_getter=_doc_retrieval_text)


def _apply_filename_boost(query_plan: QueryPlan, candidates: list[dict]) -> list[dict]:
    return _retrieval_apply_filename_boost(
        query_plan,
        candidates,
        doc_scope_match_boost=DOC_SCOPE_MATCH_BOOST,
        filename_normalizer=_qp_normalize_filename,
        milvus_rrf_k=MILVUS_RRF_K,
        filename_boost_weight=DOC_SCOPE_FILENAME_BOOST_WEIGHT,
    )


def _apply_heading_lexical_scoring(
    query_plan: QueryPlan,
    candidates: list[dict],
) -> list[dict]:
    return _retrieval_apply_heading_lexical_scoring(
        query_plan,
        candidates,
        heading_lexical_weight=HEADING_LEXICAL_WEIGHT,
        milvus_rrf_k=MILVUS_RRF_K,
    )


def _rerank_rrf_score(doc: dict) -> float:
    return _rerank_rrf_score_impl(doc, milvus_rrf_k=MILVUS_RRF_K)


def _apply_rerank_score_fusion(indexed_scores: list[tuple[int, float]], docs_for_rerank: list[dict]) -> list[tuple[int, float]]:
    weights = {
        "rerank": max(0.0, RERANK_FUSION_RERANK_WEIGHT),
        "rrf": max(0.0, RERANK_FUSION_RRF_WEIGHT),
        "scope": max(0.0, RERANK_FUSION_SCOPE_WEIGHT),
        "metadata": max(0.0, RERANK_FUSION_METADATA_WEIGHT),
    }
    return _rerank_apply_score_fusion(
        indexed_scores,
        docs_for_rerank,
        enabled=RERANK_SCORE_FUSION_ENABLED,
        weights=weights,
        milvus_rrf_k=MILVUS_RRF_K,
    )


def _finish_retrieval_pipeline(
    query: str,
    search_query: str,
    retrieved: list[dict],
    top_k: int,
    candidate_k: int,
    timings: Dict[str, float],
    stage_errors: list[StageErrorDict],
    total_start: float,
    extra_trace: dict | None = None,
    query_plan: QueryPlan | None = None,
    context_files: list[str] | None = None,
    base_filter: str | None = None,
    retrieval_mode: str = "hybrid",
    hybrid_error: str | None = None,
) -> Dict[str, Any]:
    """Complete the retrieval pipeline: rerank -> structure_rerank -> confidence_gate."""
    current_stage = "rerank"
    try:
        stage_start = time.perf_counter()
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        timings["rerank_ms"] = elapsed_ms(stage_start)
        if rerank_meta.get("rerank_error"):
            stage_errors.append(_stage_error("rerank", str(rerank_meta.get("rerank_error")), "ranked_candidates"))

        rerank_meta["ce_dtype"] = RERANK_TORCH_DTYPE
        rerank_meta["ce_input_count"] = rerank_meta.get("rerank_input_count", 0)
        rerank_meta["ce_cache_hit"] = rerank_meta.get("rerank_cache_hit", False)
        rerank_meta["ce_latency_ms"] = rerank_meta.get("ce_predict_ms", timings.get("rerank_ms", 0.0))
        rerank_meta["model_warmup_state"] = "warm" if _local_reranker is not None else "cold"

        current_stage = "structure_rerank"
        stage_start = time.perf_counter()
        reranked_docs, structure_meta = _apply_structure_rerank(docs=reranked, top_k=top_k)
        timings["structure_rerank_ms"] = elapsed_ms(stage_start)

        current_stage = "confidence_gate"
        stage_start = time.perf_counter()
        confidence_meta = _evaluate_retrieval_confidence(query=query, docs=reranked_docs)
        timings["confidence_ms"] = elapsed_ms(stage_start)
        timings["total_retrieve_ms"] = elapsed_ms(total_start)

        rerank_meta["retrieval_mode"] = retrieval_mode
        rerank_meta["candidate_k"] = candidate_k
        rerank_meta["candidate_count_before_rerank"] = len(retrieved)
        rerank_meta["candidate_count_after_rerank"] = len(reranked)
        rerank_meta["candidate_count_after_structure_rerank"] = len(reranked_docs)
        rerank_meta["milvus_search_ef"] = MILVUS_SEARCH_EF
        rerank_meta["milvus_sparse_drop_ratio"] = MILVUS_SPARSE_DROP_RATIO
        rerank_meta["milvus_rrf_k"] = MILVUS_RRF_K
        rerank_meta["leaf_retrieve_level"] = LEAF_RETRIEVE_LEVEL
        rerank_meta["index_profile"] = RAG_INDEX_PROFILE
        rerank_meta["context_files"] = context_files or []
        rerank_meta["hybrid_error"] = hybrid_error
        rerank_meta["dense_error"] = None
        rerank_meta["timings"] = dict(_ensure_retrieve_timing_defaults(timings))
        rerank_meta["stage_errors"] = stage_errors
        rerank_meta.update(structure_meta)
        rerank_meta.update(confidence_meta)
        rerank_meta["candidates_before_rerank"] = _candidate_trace(retrieved)
        rerank_meta["candidates_after_rerank"] = _candidate_trace(reranked)
        rerank_meta["candidates_after_structure_rerank"] = _candidate_trace(reranked_docs)

        # Add QueryPlan trace fields
        if extra_trace:
            rerank_meta.update(extra_trace)

        return {"docs": reranked_docs, "meta": build_retrieval_meta(rerank_meta)}

    except Exception as exc:
        stage_errors.append(_stage_error(current_stage, str(exc)))
        timings["total_retrieve_ms"] = elapsed_ms(total_start)
        return {
            "docs": [],
            "meta": build_retrieval_meta({
                "rerank_enabled": _is_rerank_enabled(),
                "rerank_applied": False,
                "rerank_model": RERANK_MODEL,
                "rerank_error": str(exc),
                "hybrid_error": hybrid_error,
                "dense_error": None,
                "retrieval_mode": "failed",
                "candidate_k": candidate_k,
                "candidate_count_before_rerank": len(retrieved),
                "candidate_count_after_rerank": 0,
                "candidate_count_after_structure_rerank": 0,
                "rerank_top_n": _effective_rerank_top_n(top_k, 0),
                "milvus_search_ef": MILVUS_SEARCH_EF,
                "milvus_sparse_drop_ratio": MILVUS_SPARSE_DROP_RATIO,
                "milvus_rrf_k": MILVUS_RRF_K,
                "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
                "index_profile": RAG_INDEX_PROFILE,
                "context_files": context_files or [],
                "structure_rerank_applied": False,
                "structure_rerank_enabled": STRUCTURE_RERANK_ENABLED,
                "structure_rerank_root_weight": STRUCTURE_RERANK_ROOT_WEIGHT,
                "same_root_cap": SAME_ROOT_CAP,
                "auto_merge_enabled": AUTO_MERGE_ENABLED,
                "auto_merge_applied": False,
                "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                "auto_merge_replaced_chunks": 0,
                "auto_merge_steps": 0,
                "candidate_count": 0,
                "confidence_gate_enabled": CONFIDENCE_GATE_ENABLED,
                "fallback_required": CONFIDENCE_GATE_ENABLED,
                "confidence_reasons": ["retrieve_failed"] if CONFIDENCE_GATE_ENABLED else [],
                "candidates_before_rerank": [],
                "candidates_after_rerank": [],
                "candidates_after_structure_rerank": [],
                "timings": dict(_ensure_retrieve_timing_defaults(timings)),
                "stage_errors": stage_errors,
                **(extra_trace or {}),
            }),
        }


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _candidate_trace(docs: List[dict]) -> List[dict]:
    traced: List[dict] = []
    for doc in docs:
        traced.append(
            {
                "chunk_id": doc.get("chunk_id", ""),
                "candidate_id": candidate_identity(doc),
                "root_chunk_id": doc.get("root_chunk_id", ""),
                "anchor_id": doc.get("anchor_id", ""),
                "filename": doc.get("filename", ""),
                "section_title": doc.get("section_title", ""),
                "section_path": doc.get("section_path", ""),
                "page_number": doc.get("page_number"),
                "page_start": doc.get("page_start"),
                "page_end": doc.get("page_end"),
                "text_hash": trace_text_hash(_doc_retrieval_text(doc)),
                "text_preview": _doc_retrieval_text(doc)[:240],
                "index_profile": doc.get("index_profile", RAG_INDEX_PROFILE),
                "score": doc.get("score"),
                "raw_rerank_score": doc.get("raw_rerank_score"),
                "rerank_score": doc.get("rerank_score"),
                "fusion_score": doc.get("fusion_score"),
                "final_score": doc.get("final_score"),
                "doc_scope_match_score": doc.get("doc_scope_match_score"),
                "filename_boost_applied": doc.get("filename_boost_applied", False),
                "filename_boost_score": doc.get("filename_boost_score"),
            }
        )
    return traced




def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _effective_candidate_k(top_k: int) -> int:
    if RAG_CANDIDATE_K > 0:
        return max(top_k, RAG_CANDIDATE_K)
    return max(top_k * 10, 50)


def _effective_rerank_top_n(top_k: int, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    requested = RERANK_TOP_N if RERANK_TOP_N > 0 else top_k
    requested = max(top_k, requested)
    return min(candidate_count, requested)


def _normalize_device_request(value: str | None, *, default: str = "auto") -> str:
    requested = (value or default).strip().lower()
    aliases = {
        "": default,
        "a1": "auto",
        "auto": "auto",
        "gpu_first": "auto",
        "gpu-first": "auto",
        "cuda_if_available": "auto",
        "a2": "cuda",
        "gpu": "cuda",
        "cuda": "cuda",
        "gpu_only": "cuda",
        "gpu-only": "cuda",
        "a0": "cpu",
        "cpu": "cpu",
        "cpu_only": "cpu",
        "cpu-only": "cpu",
    }
    return aliases.get(requested, requested)


def _resolve_rerank_device() -> str:
    requested = _normalize_device_request(RERANK_DEVICE)
    if requested == "cpu":
        return "cpu"
    if requested != "cuda" and requested != "auto":
        return requested

    import torch

    cuda_available = torch.cuda.is_available()
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA is not available but RERANK_DEVICE is set to GPU-only mode")
        return "cuda"
    return "cuda" if cuda_available else "cpu"


def _rerank_device_tier() -> str:
    return "gpu" if _resolve_rerank_device().startswith("cuda") else "cpu"


def _effective_rerank_input_k(rerank_top_n: int, candidate_count: int) -> tuple[int, str, int]:
    if candidate_count <= 0:
        return 0, _rerank_device_tier(), 0
    device_tier = _rerank_device_tier()
    cap = RERANK_INPUT_K_GPU if device_tier == "gpu" else RERANK_INPUT_K_CPU
    if cap > 0:
        return min(candidate_count, cap), device_tier, cap
    return candidate_count, device_tier, cap


def _is_rerank_enabled() -> bool:
    return bool(RERANK_MODEL)


def _milvus_index_version() -> str:
    return cache.get_string("milvus_index_version") or "0"


def _bm25_total_docs() -> int:
    try:
        return int(getattr(_embedding_service, "_total_docs", 0) or 0)
    except Exception:
        return 0


def _rerank_doc_signatures(docs: List[dict], enrichment_enabled: bool) -> List[dict]:
    signatures: List[dict] = []
    for doc in docs:
        if enrichment_enabled:
            pair_text = _build_enriched_pair(doc)
        else:
            pair_text = _doc_retrieval_text(doc)
        signatures.append(
            {
                "chunk_id": str(doc.get("chunk_id") or doc.get("id") or ""),
                "pair_text_sha1": _sha1_text(pair_text),
            }
        )
    return signatures


def _rerank_cache_key(
    query: str,
    docs_for_rerank: List[dict],
    rerank_top_n: int,
    rerank_input_k: int,
    enrichment_enabled: bool,
) -> str:
    payload = {
        "version": 2,
        "query": query,
        "provider": RERANK_PROVIDER,
        "model": RERANK_MODEL or "",
        "device_tier": _rerank_device_tier(),
        "rerank_top_n": rerank_top_n,
        "rerank_input_k": rerank_input_k,
        "score_fusion_enabled": RERANK_SCORE_FUSION_ENABLED,
        "fusion_weights": {
            "rerank": RERANK_FUSION_RERANK_WEIGHT,
            "rrf": RERANK_FUSION_RRF_WEIGHT,
            "scope": RERANK_FUSION_SCOPE_WEIGHT,
            "metadata": RERANK_FUSION_METADATA_WEIGHT,
        },
        "bm25_total_docs": _bm25_total_docs(),
        "milvus_index_version": _milvus_index_version(),
        "docs": _rerank_doc_signatures(docs_for_rerank, enrichment_enabled),
    }
    digest = _sha1_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return f"rerank:{digest}"


def _load_cached_rerank_result(cache_key: str, docs_for_rerank: List[dict], rerank_top_n: int) -> List[dict] | None:
    cached = cache.get_json(cache_key)
    if not isinstance(cached, dict):
        return None
    items = cached.get("items")
    if not isinstance(items, list):
        return None

    reranked: List[dict] = []
    for item in items[:rerank_top_n]:
        if not isinstance(item, dict):
            return None
        try:
            index = int(item["index"])
        except (KeyError, TypeError, ValueError):
            return None
        if index < 0 or index >= len(docs_for_rerank):
            return None
        doc = dict(docs_for_rerank[index])
        score = item.get("rerank_score")
        if score is not None:
            try:
                doc["rerank_score"] = float(score)
            except (TypeError, ValueError):
                return None
        raw_score = item.get("raw_rerank_score")
        if raw_score is not None:
            try:
                doc["raw_rerank_score"] = float(raw_score)
            except (TypeError, ValueError):
                return None
        fusion_score = item.get("fusion_score")
        if fusion_score is not None:
            try:
                doc["fusion_score"] = float(fusion_score)
            except (TypeError, ValueError):
                return None
        reranked.append(doc)

    return reranked if reranked else None


def _store_rerank_result(cache_key: str, reranked: List[dict], docs_for_rerank: List[dict]) -> None:
    rank_to_index = {doc.get("rrf_rank"): idx for idx, doc in enumerate(docs_for_rerank)}
    items = []
    for doc in reranked:
        index = rank_to_index.get(doc.get("rrf_rank"))
        if index is None:
            return
        items.append(
            {
                "index": index,
                "rerank_score": doc.get("rerank_score"),
                "raw_rerank_score": doc.get("raw_rerank_score"),
                "fusion_score": doc.get("fusion_score"),
            }
        )
    cache.set_json(cache_key, {"items": items}, ttl=RERANK_CACHE_TTL_SECONDS)


def _get_local_reranker():
    global _local_reranker
    if not RERANK_MODEL:
        return None
    if _local_reranker is None:
        import torch
        from sentence_transformers import CrossEncoder
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(RERANK_TORCH_DTYPE.lower(), torch.float16)
        device = _resolve_rerank_device()
        _local_reranker = CrossEncoder(RERANK_MODEL, device=device, model_kwargs={"torch_dtype": dtype})
    return _local_reranker


_RETRIEVE_TIMING_KEYS = (
    "embed_dense_ms",
    "embed_sparse_ms",
    "milvus_hybrid_ms",
    "milvus_dense_fallback_ms",
    "l1_prefilter_ms",
    "rerank_ms",
    "structure_rerank_ms",
    "confidence_ms",
    "total_retrieve_ms",
)


def _ensure_retrieve_timing_defaults(timings: Dict[str, float]) -> Dict[str, float]:
    for key in _RETRIEVE_TIMING_KEYS:
        timings.setdefault(key, 0.0)
    return timings


def _stage_error(
    stage: str,
    error: str,
    fallback_to: str | None = None,
    *,
    error_code: str | None = None,
    severity: str = "warning",
    recoverable: bool = True,
    user_visible: bool = False,
) -> dict[str, Any]:
    return StageError(
        stage=stage,
        error=error,
        fallback_to=fallback_to,
        error_code=error_code,
        severity=severity,
        recoverable=recoverable,
        user_visible=user_visible,
    ).as_dict()


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    return _context_merge_to_parent_level(
        docs,
        parent_store_getter=_get_parent_chunk_store,
        threshold=threshold,
    )


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    return _context_auto_merge_documents(
        docs,
        top_k,
        enabled=AUTO_MERGE_ENABLED,
        threshold=AUTO_MERGE_THRESHOLD,
        parent_store_getter=_get_parent_chunk_store,
    )


def _apply_structure_rerank(docs: List[dict], top_k: int, root_weight: float | None = None, same_root_cap: int | None = None) -> Tuple[List[dict], Dict[str, Any]]:
    return _context_apply_structure_rerank(
        docs,
        top_k,
        enabled=STRUCTURE_RERANK_ENABLED,
        root_weight=root_weight if root_weight is not None else STRUCTURE_RERANK_ROOT_WEIGHT,
        same_root_cap=same_root_cap if same_root_cap is not None else SAME_ROOT_CAP,
    )


def _extract_query_anchors(query: str) -> list[str]:
    return _confidence_extract_query_anchors(query, anchor_pattern=_ANCHOR_PATTERN)


def _doc_matches_anchor(doc: dict, anchor: str) -> bool:
    return _confidence_doc_matches_anchor(doc, anchor, doc_text_getter=_doc_retrieval_text)


def _evaluate_retrieval_confidence(query: str, docs: List[dict]) -> Dict[str, Any]:
    return _confidence_evaluate_retrieval_confidence(
        query,
        docs,
        confidence_gate_enabled=CONFIDENCE_GATE_ENABLED,
        low_conf_top_margin=LOW_CONF_TOP_MARGIN,
        low_conf_root_share=LOW_CONF_ROOT_SHARE,
        low_conf_top_score=LOW_CONF_TOP_SCORE,
        enable_anchor_gate=ENABLE_ANCHOR_GATE,
        anchor_pattern=_ANCHOR_PATTERN,
        doc_text_getter=_doc_retrieval_text,
    )


def _rerank_pair_text(doc: dict, enrichment_enabled: bool) -> str:
    return _rerank_pair_text_impl(doc, enrichment_enabled, doc_text_getter=_doc_retrieval_text)


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    runtime = RerankRuntime(
        provider=RERANK_PROVIDER,
        model=RERANK_MODEL,
        cpu_top_n_cap=RERANK_INPUT_K_CPU,
        cache_enabled=RERANK_CACHE_ENABLED,
        pair_enrichment_enabled=bool(RERANK_PAIR_ENRICHMENT_ENABLED),
        score_fusion_enabled=RERANK_SCORE_FUSION_ENABLED,
        fusion_weights={
            "rerank": RERANK_FUSION_RERANK_WEIGHT,
            "rrf": RERANK_FUSION_RRF_WEIGHT,
            "scope": RERANK_FUSION_SCOPE_WEIGHT,
            "metadata": RERANK_FUSION_METADATA_WEIGHT,
        },
        milvus_rrf_k=MILVUS_RRF_K,
        effective_top_n=_effective_rerank_top_n,
        effective_input_k=_effective_rerank_input_k,
        get_local_reranker=_get_local_reranker,
        cache_key=_rerank_cache_key,
        load_cached_result=_load_cached_rerank_result,
        store_result=_store_rerank_result,
        doc_text_getter=_doc_retrieval_text,
    )
    return _run_rerank_documents(query, docs, top_k, runtime)


def _get_stepback_model():
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        from langchain.chat_models import init_chat_model

        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
        )
    return _stepback_model


def _generate_step_back_pair(query: str) -> tuple[str, str]:
    model = _get_stepback_model()
    if not model:
        return "", ""
    prompt = (
        "Generate one step-back question and one concise answer for retrieval expansion.\n"
        "Return JSON only with this exact schema:\n"
        "{\"step_back_question\":\"...\",\"step_back_answer\":\"...\"}\n"
        "Rules:\n"
        "- step_back_question abstracts the user's concrete question into a broader principle question.\n"
        "- step_back_answer answers that broader question in no more than 120 Chinese characters when possible.\n"
        "- Do not include reasoning, markdown, or extra keys.\n"
        f"User question: {query}"
    )
    try:
        response = model.invoke(prompt)
        payload = extract_json_object(getattr(response, "content", response) or "")
        step_back_question = str(payload.get("step_back_question") or "").strip()
        step_back_answer = str(payload.get("step_back_answer") or "").strip()
        return step_back_question, step_back_answer
    except Exception:
        return "", ""


def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    step_back_question, step_back_answer = _generate_step_back_pair(query)
    if step_back_question or step_back_answer:
        expanded_query = (
            f"{query}\n\n"
            f"退步问题：{step_back_question}\n"
            f"退步问题答案：{step_back_answer}"
        )
    else:
        expanded_query = query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _build_filename_filter(filenames: list[str] | None) -> str:
    return _retrieval_build_filename_filter(filenames, escape_string=_escape_milvus_string)


def build_query_plan(
    request: RetrievalRequest,
    timings: Dict[str, float] | None = None,
    stage_errors: list[StageErrorDict] | None = None,
) -> QueryPlan:
    stage_start = time.perf_counter()
    try:
        if QUERY_PLAN_ENABLED:
            filename_registry = get_filename_registry(_milvus_manager, cache)
            return parse_query_plan(
                raw_query=request.query,
                filename_registry=filename_registry,
                context_files=request.context_files,
            )
    except Exception as exc:
        if stage_errors is not None:
            stage_errors.append(_stage_error("query_plan", str(exc), "raw_query"))
    finally:
        if timings is not None:
            timings["query_plan_ms"] = elapsed_ms(stage_start)

    return QueryPlan(
        raw_query=request.query,
        semantic_query=request.query,
        clean_query=request.query,
        doc_hints=[],
        matched_files=[],
        scope_mode="none",
        heading_hint=None,
        anchors=[],
        model_numbers=[],
        intent_type=None,
        route="global_hybrid",
    )


def build_retrieval_filters(query_plan: QueryPlan, context_files: list[str] | None = None) -> RetrievalFilters:
    base_filter = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    filename_filter = _build_filename_filter(context_files)
    effective_filter = f"{base_filter} and {filename_filter}" if filename_filter else base_filter
    matched_files = [
        (filename, score)
        for filename, score in query_plan.matched_files
        if score >= DOC_SCOPE_MATCH_BOOST
    ]
    scoped_filter = None
    if matched_files and not filename_filter:
        scoped_filename_filter = _build_filename_filter([filename for filename, _ in matched_files])
        if scoped_filename_filter:
            scoped_filter = f"{base_filter} and {scoped_filename_filter}"
    return RetrievalFilters(
        base_filter=base_filter,
        filename_filter=filename_filter,
        effective_filter=effective_filter,
        scoped_filter=scoped_filter,
        matched_files=matched_files,
    )


def embed_search_query(
    search_query: str,
    timings: Dict[str, float],
    stage_errors: list[StageErrorDict],
) -> QueryEmbeddings:
    stage_start = time.perf_counter()
    dense_embedding = None
    sparse_embedding = None
    dense_error = None
    sparse_error = None
    with ThreadPoolExecutor(max_workers=2) as embed_pool:
        dense_future = embed_pool.submit(_embedding_service.get_embeddings, [search_query])
        sparse_future = embed_pool.submit(_embedding_service.get_sparse_embedding, search_query)
        try:
            dense_embedding = dense_future.result()[0]
        except Exception as exc:
            dense_error = str(exc)
            stage_errors.append(_stage_error("embed_dense", dense_error, "failed"))
        try:
            sparse_embedding = sparse_future.result()
        except Exception as exc:
            sparse_error = str(exc)
            stage_errors.append(_stage_error("embed_sparse", sparse_error, "dense_retrieve"))
    timings["embed_dense_ms"] = elapsed_ms(stage_start)
    timings["embed_sparse_ms"] = 0.0
    return QueryEmbeddings(
        dense=dense_embedding,
        sparse=sparse_embedding,
        dense_error=dense_error,
        sparse_error=sparse_error,
    )


def _query_plan_trace(
    query_plan: QueryPlan,
    *,
    semantic_query: str,
    v3_layers: list[str],
    scope_filter_applied: bool = False,
) -> dict[str, Any]:
    return {
        "query_plan": query_plan.to_dict(),
        "query_plan_enabled": QUERY_PLAN_ENABLED,
        "semantic_query": semantic_query,
        "index_profile": RAG_INDEX_PROFILE,
        "v3_layers": v3_layers,
        "scope_mode": query_plan.scope_mode,
        "query_route": query_plan.route,
        "scope_filter_applied": scope_filter_applied,
        "filename_boost_applied": False,
        "filename_boosted_candidate_count": 0,
        "matched_files_top3": [(f, round(s, 3)) for f, s in query_plan.matched_files[:3]],
        "doc_scope_match_ratios": [round(s, 3) for _, s in query_plan.matched_files[:3]],
    }


def _with_candidate_strategy(
    trace_patch: dict[str, Any] | None,
    strategy: CandidateStrategy,
    family: CandidateStrategyFamily,
    *,
    fallback_from: CandidateStrategy | None = None,
) -> dict[str, Any]:
    patch = dict(trace_patch or {})
    patch.update(candidate_strategy_trace(strategy, family, fallback_from=fallback_from))
    return patch


def _append_hybrid_guarantee(candidates: list[dict], hybrid: list[dict]) -> None:
    existing_ids = {candidate.get("chunk_id") for candidate in candidates}
    for item in hybrid:
        if item.get("chunk_id") in existing_ids:
            continue
        item["dense_rank"] = None
        item["sparse_rank"] = None
        item["dense_score"] = 0.0
        item["sparse_score"] = 0.0
        item["in_dense"] = False
        item["in_sparse"] = False
        candidates.append(item)
        existing_ids.add(item.get("chunk_id"))


def _dense_candidate_result(
    embeddings: QueryEmbeddings,
    *,
    candidate_k: int,
    filter_expr: str,
    timings: Dict[str, float],
    retrieval_mode: str = "dense_fallback",
    trace_patch: dict[str, Any] | None = None,
    hybrid_error: str | None = None,
    candidate_strategy: CandidateStrategy = CandidateStrategy.DENSE_FALLBACK,
    candidate_strategy_family: CandidateStrategyFamily = CandidateStrategyFamily.STANDARD,
    candidate_strategy_fallback_from: CandidateStrategy | None = None,
) -> CandidateRetrievalResult:
    strategy_trace = _with_candidate_strategy(
        trace_patch,
        candidate_strategy,
        candidate_strategy_family,
        fallback_from=candidate_strategy_fallback_from,
    )
    if embeddings.dense is None:
        dense_error = embeddings.dense_error or "dense embedding unavailable"
        return CandidateRetrievalResult(
            candidates=[],
            retrieval_mode="failed",
            trace_patch=strategy_trace,
            stage_errors=[_stage_error("embed_dense", dense_error)],
            hybrid_error=hybrid_error,
        )
    stage_start = time.perf_counter()
    try:
        candidates = _milvus_manager.dense_retrieve(
            dense_embedding=embeddings.dense,
            top_k=candidate_k,
            search_ef=MILVUS_SEARCH_EF,
            filter_expr=filter_expr,
        )
    except Exception as exc:
        timings["milvus_dense_fallback_ms"] = elapsed_ms(stage_start)
        return CandidateRetrievalResult(
            candidates=[],
            retrieval_mode="failed",
            trace_patch=strategy_trace,
            stage_errors=[_stage_error("dense_retrieve", str(exc))],
            hybrid_error=hybrid_error,
        )
    timings["milvus_dense_fallback_ms"] = elapsed_ms(stage_start)
    return CandidateRetrievalResult(
        candidates=candidates,
        retrieval_mode=retrieval_mode,
        trace_patch=strategy_trace,
        hybrid_error=hybrid_error,
    )


def retrieve_global_candidates(
    embeddings: QueryEmbeddings,
    *,
    candidate_k: int,
    filter_expr: str,
    timings: Dict[str, float],
    trace_patch: dict[str, Any],
    candidate_strategy: CandidateStrategy = CandidateStrategy.GLOBAL_HYBRID,
) -> CandidateRetrievalResult:
    strategy_trace = _with_candidate_strategy(
        trace_patch,
        candidate_strategy,
        CandidateStrategyFamily.STANDARD,
    )
    if embeddings.dense is None:
        return CandidateRetrievalResult(
            candidates=[],
            retrieval_mode="failed",
            trace_patch=strategy_trace,
            stage_errors=[_stage_error("embed_dense", embeddings.dense_error or "dense embedding unavailable")],
        )
    if embeddings.sparse is None:
        return _dense_candidate_result(
            embeddings,
            candidate_k=candidate_k,
            filter_expr=filter_expr,
            timings=timings,
            retrieval_mode="dense_fallback",
            trace_patch=trace_patch,
            candidate_strategy_fallback_from=candidate_strategy,
        )

    stage_start = time.perf_counter()
    try:
        candidates = _milvus_manager.hybrid_retrieve(
            dense_embedding=embeddings.dense,
            sparse_embedding=embeddings.sparse,
            top_k=candidate_k,
            rrf_k=MILVUS_RRF_K,
            search_ef=MILVUS_SEARCH_EF,
            sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
            filter_expr=filter_expr,
        )
        timings["milvus_hybrid_ms"] = elapsed_ms(stage_start)
        return CandidateRetrievalResult(candidates=candidates, retrieval_mode="hybrid", trace_patch=strategy_trace)
    except Exception as exc:
        hybrid_error = str(exc)
        timings["milvus_hybrid_ms"] = elapsed_ms(stage_start)
        try:
            result = _dense_candidate_result(
                embeddings,
                candidate_k=candidate_k,
                filter_expr=filter_expr,
                timings=timings,
                trace_patch=trace_patch,
                hybrid_error=hybrid_error,
                candidate_strategy_fallback_from=candidate_strategy,
            )
            result.stage_errors = [_stage_error("hybrid_retrieve", hybrid_error, "dense_retrieve")] + result.stage_errors
            return result
        except Exception as dense_exc:
            return CandidateRetrievalResult(
                candidates=[],
                retrieval_mode="failed",
                trace_patch=strategy_trace,
                stage_errors=[
                    _stage_error("hybrid_retrieve", hybrid_error, "dense_retrieve"),
                    _stage_error("dense_retrieve", str(dense_exc)),
                ],
                hybrid_error=hybrid_error,
            )


def retrieve_scoped_candidates(
    embeddings: QueryEmbeddings,
    *,
    candidate_k: int,
    filters: RetrievalFilters,
    timings: Dict[str, float],
    trace_patch: dict[str, Any],
) -> CandidateRetrievalResult:
    strategy_trace = _with_candidate_strategy(
        trace_patch,
        CandidateStrategy.SCOPED_HYBRID,
        CandidateStrategyFamily.STANDARD,
    )
    if embeddings.dense is None:
        return CandidateRetrievalResult(
            candidates=[],
            retrieval_mode="failed",
            trace_patch=strategy_trace,
            stage_errors=[_stage_error("embed_dense", embeddings.dense_error or "dense embedding unavailable")],
        )
    if embeddings.sparse is None:
        return _dense_candidate_result(
            embeddings,
            candidate_k=candidate_k,
            filter_expr=filters.scoped_filter or filters.effective_filter,
            timings=timings,
            retrieval_mode="dense_fallback_scoped",
            trace_patch=trace_patch,
            candidate_strategy_fallback_from=CandidateStrategy.SCOPED_HYBRID,
        )

    stage_start = time.perf_counter()

    def _scoped_retrieve():
        return _milvus_manager.hybrid_retrieve(
            dense_embedding=embeddings.dense,
            sparse_embedding=embeddings.sparse,
            top_k=candidate_k,
            rrf_k=MILVUS_RRF_K,
            search_ef=MILVUS_SEARCH_EF,
            sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
            filter_expr=filters.scoped_filter,
        )

    def _global_retrieve():
        return _milvus_manager.hybrid_retrieve(
            dense_embedding=embeddings.dense,
            sparse_embedding=embeddings.sparse,
            top_k=candidate_k,
            rrf_k=MILVUS_RRF_K,
            search_ef=MILVUS_SEARCH_EF,
            sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
            filter_expr=filters.base_filter,
        )

    stage_errors: list[StageErrorDict] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        scoped_future = pool.submit(_scoped_retrieve)
        global_future = pool.submit(_global_retrieve)
        try:
            scoped = scoped_future.result()
        except Exception as exc:
            stage_errors.append(_stage_error("scoped_retrieve", str(exc), "global_only"))
            scoped = []
        try:
            global_ = global_future.result()
        except Exception as exc:
            stage_errors.append(_stage_error("global_retrieve", str(exc), "scoped_only"))
            global_ = []

    timings["milvus_hybrid_ms"] = elapsed_ms(stage_start)
    if not scoped and not global_:
        try:
            dense = _dense_candidate_result(
                embeddings,
                candidate_k=candidate_k,
                filter_expr=filters.scoped_filter or filters.effective_filter,
                timings=timings,
                retrieval_mode="dense_fallback_scoped",
                trace_patch=trace_patch,
                candidate_strategy_fallback_from=CandidateStrategy.SCOPED_HYBRID,
            )
            dense.stage_errors.extend(stage_errors)
            return dense
        except Exception as exc:
            stage_errors.append(_stage_error("dense_retrieve", str(exc)))
            return CandidateRetrievalResult([], "failed", trace_patch, stage_errors)

    scoped = _retrieval_annotate_scope_scores(scoped, filters.matched_files)
    global_ = _retrieval_annotate_scope_scores(global_, filters.matched_files)
    merged = _retrieval_weighted_rrf_merge(
        [(scoped, 1.0 - DOC_SCOPE_GLOBAL_RESERVE_WEIGHT), (global_, DOC_SCOPE_GLOBAL_RESERVE_WEIGHT)],
        rrf_k=MILVUS_RRF_K,
    )
    patch = dict(trace_patch)
    patch.update(strategy_trace)
    patch["scoped_candidate_count"] = len(scoped)
    patch["global_candidate_count"] = len(global_)
    return CandidateRetrievalResult(merged, "hybrid_scoped", patch, stage_errors)


def retrieve_layered_candidates(
    embeddings: QueryEmbeddings,
    *,
    candidate_k: int,
    filter_expr: str,
    query_plan: QueryPlan,
    scope_matched_files: list[tuple[str, float]],
    timings: Dict[str, float],
    trace_patch: dict[str, Any],
    retrieval_mode: str,
) -> CandidateRetrievalResult:
    layered_trace = _with_candidate_strategy(
        trace_patch,
        CandidateStrategy.LAYERED_SPLIT,
        CandidateStrategyFamily.LAYERED,
    )
    if embeddings.dense is None:
        return CandidateRetrievalResult(
            candidates=[],
            retrieval_mode="failed",
            trace_patch=layered_trace,
            stage_errors=[_stage_error("embed_dense", embeddings.dense_error or "dense embedding unavailable")],
        )
    if embeddings.sparse is None:
        return _dense_candidate_result(
            embeddings,
            candidate_k=candidate_k,
            filter_expr=filter_expr,
            timings=timings,
            retrieval_mode="dense_fallback_scoped" if retrieval_mode == "hybrid_scoped" else "dense_fallback",
            trace_patch=trace_patch,
            candidate_strategy_fallback_from=CandidateStrategy.LAYERED_SPLIT,
        )

    stage_errors: list[StageErrorDict] = []
    stage_start = time.perf_counter()
    try:
        candidates = _milvus_manager.split_retrieve(
            embeddings.dense,
            embeddings.sparse,
            dense_top_k=_LAYERED_CONFIG.l0_dense_top_k,
            sparse_top_k=_LAYERED_CONFIG.l0_sparse_top_k,
            search_ef=MILVUS_SEARCH_EF,
            sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
            filter_expr=filter_expr,
        )
        if _LAYERED_CONFIG.l0_hybrid_guarantee_k > 0:
            hybrid = _milvus_manager.hybrid_retrieve(
                embeddings.dense,
                embeddings.sparse,
                top_k=_LAYERED_CONFIG.l0_hybrid_guarantee_k,
                rrf_k=MILVUS_RRF_K,
                search_ef=MILVUS_SEARCH_EF,
                sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
                filter_expr=filter_expr,
            )
            _append_hybrid_guarantee(candidates, hybrid)
        if len(candidates) < _LAYERED_CONFIG.l0_fallback_pool_min:
            hybrid = _milvus_manager.hybrid_retrieve(
                embeddings.dense,
                embeddings.sparse,
                top_k=_LAYERED_CONFIG.l0_dense_top_k,
                rrf_k=MILVUS_RRF_K,
                search_ef=MILVUS_SEARCH_EF,
                sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
                filter_expr=filter_expr,
            )
            _append_hybrid_guarantee(candidates, hybrid)
        timings["milvus_hybrid_ms"] = elapsed_ms(stage_start)
    except Exception as exc:
        stage_errors.append(_stage_error("layered_retrieve", str(exc), "standard_hybrid"))
        fallback_strategy = CandidateStrategy.SCOPED_HYBRID if retrieval_mode == "hybrid_scoped" else CandidateStrategy.GLOBAL_HYBRID
        fallback = retrieve_global_candidates(
            embeddings,
            candidate_k=candidate_k,
            filter_expr=filter_expr,
            timings=timings,
            trace_patch=trace_patch,
            candidate_strategy=fallback_strategy,
        )
        fallback.trace_patch["candidate_strategy_fallback_from"] = CandidateStrategy.LAYERED_SPLIT.value
        if fallback.retrieval_mode == "hybrid" and retrieval_mode == "hybrid_scoped":
            fallback.retrieval_mode = "hybrid_scoped"
        elif fallback.retrieval_mode == "dense_fallback" and retrieval_mode == "hybrid_scoped":
            fallback.retrieval_mode = "dense_fallback_scoped"
        fallback.stage_errors = stage_errors + fallback.stage_errors
        return fallback

    if scope_matched_files:
        candidates = _retrieval_annotate_scope_scores(candidates, scope_matched_files)
    l1_start = time.perf_counter()
    l0_candidate_count = len(candidates)
    candidates = _build_l1_candidates(
        candidates,
        scope_matched_files=[filename for filename, _ in scope_matched_files],
        anchor_chunk_ids=[doc.get("chunk_id", "") for doc in candidates if doc.get("anchor_id")],
        config=_LAYERED_CONFIG,
    )
    timings["l1_prefilter_ms"] = elapsed_ms(l1_start)
    patch = dict(layered_trace)
    patch.update({
        "v3_layers": ["query_plan", "layered_split", "l1_prefilter", "rerank", "structure_rerank"],
        "layered_l0_candidate_count": l0_candidate_count,
        "layered_candidate_count": len(candidates),
        "l1_candidate_count": len(candidates),
    })
    return CandidateRetrievalResult(candidates, retrieval_mode, patch, stage_errors)


def apply_candidate_adjustments(
    query_plan: QueryPlan,
    candidates: list[dict],
    trace_patch: dict[str, Any],
) -> tuple[list[dict], dict[str, Any]]:
    patch = dict(trace_patch)
    adjusted = candidates
    if QUERY_PLAN_ENABLED and query_plan.scope_mode == "boost":
        adjusted = _apply_filename_boost(query_plan, adjusted)
        boosted_count = sum(1 for doc in adjusted if doc.get("filename_boost_applied"))
        patch["filename_boost_applied"] = boosted_count > 0
        patch["filename_boosted_candidate_count"] = boosted_count
    if QUERY_PLAN_ENABLED and HEADING_LEXICAL_ENABLED and query_plan.scope_mode in {"filter", "boost"} and query_plan.heading_hint:
        adjusted = _apply_heading_lexical_scoring(query_plan=query_plan, candidates=adjusted)
    return adjusted, patch


def finish_retrieval_pipeline(*args, **kwargs) -> Dict[str, Any]:
    return _finish_retrieval_pipeline(*args, **kwargs)


def _failed_retrieval_response(
    *,
    top_k: int,
    candidate_k: int,
    timings: Dict[str, float],
    stage_errors: list[StageErrorDict],
    total_start: float,
    context_files: list[str] | None,
    trace_patch: dict[str, Any],
    hybrid_error: str | None = None,
    dense_error: str | None = None,
) -> Dict[str, Any]:
    timings["total_retrieve_ms"] = elapsed_ms(total_start)
    return {
        "docs": [],
        "meta": build_retrieval_meta({
            "rerank_enabled": _is_rerank_enabled(),
            "rerank_applied": False,
            "rerank_model": RERANK_MODEL,
            "rerank_error": "retrieve_failed",
            "hybrid_error": hybrid_error,
            "dense_error": dense_error,
            "retrieval_mode": "failed",
            "candidate_k": candidate_k,
            "candidate_count_before_rerank": 0,
            "candidate_count_after_rerank": 0,
            "candidate_count_after_structure_rerank": 0,
            "rerank_top_n": _effective_rerank_top_n(top_k, 0),
            "milvus_search_ef": MILVUS_SEARCH_EF,
            "milvus_sparse_drop_ratio": MILVUS_SPARSE_DROP_RATIO,
            "milvus_rrf_k": MILVUS_RRF_K,
            "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
            "index_profile": RAG_INDEX_PROFILE,
            "context_files": context_files or [],
            "structure_rerank_applied": False,
            "structure_rerank_enabled": STRUCTURE_RERANK_ENABLED,
            "structure_rerank_root_weight": STRUCTURE_RERANK_ROOT_WEIGHT,
            "same_root_cap": SAME_ROOT_CAP,
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
            "candidate_count": 0,
            "confidence_gate_enabled": CONFIDENCE_GATE_ENABLED,
            "fallback_required": CONFIDENCE_GATE_ENABLED,
            "confidence_reasons": ["retrieve_failed"] if CONFIDENCE_GATE_ENABLED else [],
            "candidates_before_rerank": [],
            "candidates_after_rerank": [],
            "candidates_after_structure_rerank": [],
            "timings": dict(_ensure_retrieve_timing_defaults(timings)),
            "stage_errors": stage_errors,
            **trace_patch,
        }),
    }


def prepare_candidate_retrieval(
    query: str,
    top_k: int = 5,
    context_files: list[str] | None = None,
    *,
    candidate_k_override: int | None = None,
) -> PreparedRetrieval:
    total_start = time.perf_counter()
    timings: Dict[str, float] = {}
    stage_errors: list[StageErrorDict] = []
    request = RetrievalRequest(query=query, top_k=top_k, context_files=list(context_files or []))
    candidate_k = candidate_k_override if candidate_k_override is not None else _effective_candidate_k(top_k)

    query_plan = build_query_plan(request, timings, stage_errors)
    search_query = query_plan.semantic_query
    filters = build_retrieval_filters(query_plan, request.context_files)
    embeddings = embed_search_query(search_query, timings, stage_errors)

    use_scoped = (
        QUERY_PLAN_ENABLED
        and query_plan.scope_mode == "filter"
        and bool(filters.matched_files)
        and not filters.filename_filter
        and bool(filters.scoped_filter)
    )
    if use_scoped:
        trace_patch = _query_plan_trace(
            query_plan,
            semantic_query=search_query,
            v3_layers=["query_plan", "doc_resolver", "scoped_hybrid", "weighted_rrf", "rerank", "structure_rerank"],
            scope_filter_applied=True,
        )
        if _LAYERED_CONFIG.enabled:
            result = retrieve_layered_candidates(
                embeddings,
                candidate_k=candidate_k,
                filter_expr=filters.scoped_filter or filters.effective_filter,
                query_plan=query_plan,
                scope_matched_files=filters.matched_files,
                timings=timings,
                trace_patch=trace_patch,
                retrieval_mode="hybrid_scoped",
            )
        else:
            result = retrieve_scoped_candidates(
                embeddings,
                candidate_k=candidate_k,
                filters=filters,
                timings=timings,
                trace_patch=trace_patch,
            )
    else:
        trace_patch = _query_plan_trace(
            query_plan,
            semantic_query=search_query,
            v3_layers=["query_plan", "global_hybrid", "rerank", "structure_rerank"],
            scope_filter_applied=False,
        )
        if _LAYERED_CONFIG.enabled:
            result = retrieve_layered_candidates(
                embeddings,
                candidate_k=candidate_k,
                filter_expr=filters.effective_filter,
                query_plan=query_plan,
                scope_matched_files=filters.matched_files,
                timings=timings,
                trace_patch=trace_patch,
                retrieval_mode="hybrid",
            )
        else:
            result = retrieve_global_candidates(
                embeddings,
                candidate_k=candidate_k,
                filter_expr=filters.effective_filter,
                timings=timings,
                trace_patch=trace_patch,
            )

    stage_errors.extend(result.stage_errors)
    retrieved, trace_patch = apply_candidate_adjustments(query_plan, result.candidates, result.trace_patch)
    retrieval_mode = result.retrieval_mode
    if trace_patch.get("filename_boost_applied") and retrieval_mode == "hybrid":
        retrieval_mode = "hybrid_boosted"
    elif trace_patch.get("filename_boost_applied") and retrieval_mode == "dense_fallback":
        retrieval_mode = "dense_fallback_boosted"

    dense_error = None
    if retrieval_mode == "failed":
        for item in stage_errors:
            if item.get("stage") in {"dense_retrieve", "embed_dense"}:
                dense_error = item.get("error")

    return PreparedRetrieval(
        query=query,
        search_query=search_query,
        top_k=top_k,
        candidate_k=candidate_k,
        context_files=request.context_files,
        query_plan=query_plan,
        filters=filters,
        candidates=retrieved,
        retrieval_mode=retrieval_mode,
        trace_patch=trace_patch,
        timings=timings,
        stage_errors=stage_errors,
        total_start=total_start,
        hybrid_error=result.hybrid_error,
        dense_error=dense_error,
    )


def retrieve_candidate_pool(
    query: str,
    top_k: int = 5,
    context_files: list[str] | None = None,
    *,
    candidate_k: int | None = None,
) -> Dict[str, Any]:
    prepared = prepare_candidate_retrieval(
        query,
        top_k=top_k,
        context_files=context_files,
        candidate_k_override=candidate_k,
    )
    prepared.timings["total_retrieve_ms"] = elapsed_ms(prepared.total_start)
    meta = build_retrieval_meta(
        {
            "retrieval_mode": prepared.retrieval_mode,
            "candidate_k": prepared.candidate_k,
            "candidate_only": True,
            "candidate_count": len(prepared.candidates),
            "candidate_count_before_rerank": len(prepared.candidates),
            "candidate_count_after_rerank": 0,
            "candidate_count_after_structure_rerank": 0,
            "rerank_enabled": _is_rerank_enabled(),
            "rerank_applied": False,
            "rerank_model": RERANK_MODEL,
            "rerank_error": None,
            "hybrid_error": prepared.hybrid_error,
            "dense_error": prepared.dense_error,
            "milvus_search_ef": MILVUS_SEARCH_EF,
            "milvus_sparse_drop_ratio": MILVUS_SPARSE_DROP_RATIO,
            "milvus_rrf_k": MILVUS_RRF_K,
            "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
            "index_profile": RAG_INDEX_PROFILE,
            "context_files": prepared.context_files,
            "candidates_before_rerank": _candidate_trace(prepared.candidates),
            "candidates_after_rerank": [],
            "candidates_after_structure_rerank": [],
            "timings": dict(_ensure_retrieve_timing_defaults(prepared.timings)),
            "stage_errors": prepared.stage_errors,
            **prepared.trace_patch,
        }
    )
    return {"candidates": prepared.candidates, "meta": meta}


def retrieve_context_documents(filenames: list[str] | None, limit_per_file: int = 8) -> Dict[str, Any]:
    """Fetch representative indexed chunks directly from attached filenames."""
    clean_files = []
    seen = set()
    for filename in filenames or []:
        name = (filename or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        clean_files.append(name)

    docs = []
    for filename in clean_files:
        filename_filter = _build_filename_filter([filename])
        if not filename_filter:
            continue
        rows = _milvus_manager.query(
            filter_expr=f"chunk_level == {LEAF_RETRIEVE_LEVEL} and {filename_filter}",
            output_fields=[
                "text",
                "retrieval_text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_role",
                "section_title",
                "section_path",
                "anchor_id",
                "chunk_idx",
            ],
            limit=limit_per_file,
        )
        rows.sort(key=lambda item: (item.get("page_number", 0), item.get("chunk_idx", 0)))
        docs.extend(rows)

    docs = _retrieval_dedupe_docs(docs)
    for idx, item in enumerate(docs, 1):
        item["rrf_rank"] = idx
    return {
        "docs": docs,
        "meta": {
            "retrieval_mode": "attached_file_context",
            "candidate_k": limit_per_file * max(len(clean_files), 1),
            "context_files": clean_files,
            "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
            "index_profile": RAG_INDEX_PROFILE,
            "attached_context_count": len(docs),
        },
    }


def retrieve_documents(query: str, top_k: int = 5, context_files: list[str] | None = None) -> Dict[str, Any]:
    prepared = prepare_candidate_retrieval(query, top_k=top_k, context_files=context_files)

    if prepared.failed:
        return _failed_retrieval_response(
            top_k=top_k,
            candidate_k=prepared.candidate_k,
            timings=prepared.timings,
            stage_errors=prepared.stage_errors,
            total_start=prepared.total_start,
            context_files=prepared.context_files,
            trace_patch=prepared.trace_patch,
            hybrid_error=prepared.hybrid_error,
            dense_error=prepared.dense_error,
        )

    return finish_retrieval_pipeline(
        query=query,
        search_query=prepared.search_query,
        retrieved=prepared.candidates,
        top_k=top_k,
        candidate_k=prepared.candidate_k,
        timings=prepared.timings,
        stage_errors=prepared.stage_errors,
        total_start=prepared.total_start,
        extra_trace=prepared.trace_patch,
        query_plan=prepared.query_plan,
        context_files=prepared.context_files,
        base_filter=prepared.filters.base_filter,
        retrieval_mode=prepared.retrieval_mode,
        hybrid_error=prepared.hybrid_error,
    )
