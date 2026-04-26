from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any
import os
import json
import hashlib
import re
import time
import requests
from dotenv import load_dotenv

from cache import cache
from json_utils import extract_json_object
from milvus_client import MilvusManager
from embedding import embedding_service as _embedding_service
from query_plan import (
    QueryPlan,
    DOC_SCOPE_MATCH_BOOST,
    parse_query_plan,
    get_filename_registry,
    _normalize_filename as _qp_normalize_filename,
)
from rag_profiles import current_index_profile
from rag_confidence import (
    doc_matches_anchor as _confidence_doc_matches_anchor,
    evaluate_retrieval_confidence as _confidence_evaluate_retrieval_confidence,
    extract_query_anchors as _confidence_extract_query_anchors,
)
from rag_context import (
    apply_structure_rerank as _context_apply_structure_rerank,
    auto_merge_documents as _context_auto_merge_documents,
    merge_to_parent_level as _context_merge_to_parent_level,
)
from rag_rerank import (
    RerankRuntime,
    apply_rerank_score_fusion as _rerank_apply_score_fusion,
    build_enriched_pair as _rerank_build_enriched_pair,
    rerank_documents as _run_rerank_documents,
    rerank_pair_text as _rerank_pair_text_impl,
    rerank_rrf_score as _rerank_rrf_score_impl,
)
from rag_retrieval import (
    annotate_scope_scores as _retrieval_annotate_scope_scores,
    apply_filename_boost as _retrieval_apply_filename_boost,
    apply_heading_lexical_scoring as _retrieval_apply_heading_lexical_scoring,
    build_filename_filter as _retrieval_build_filename_filter,
    dedupe_docs as _retrieval_dedupe_docs,
    weighted_rrf_merge as _retrieval_weighted_rrf_merge,
)
from rag_trace import candidate_identity, trace_text_hash
from rag_types import StageError

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"

ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "api").lower()
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "auto")
AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "true").lower() != "false"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))
STRUCTURE_RERANK_ROOT_WEIGHT = float(os.getenv("STRUCTURE_RERANK_ROOT_WEIGHT", "0.3"))
SAME_ROOT_CAP = int(os.getenv("SAME_ROOT_CAP", "2"))
STRUCTURE_RERANK_ENABLED = os.getenv("STRUCTURE_RERANK_ENABLED", "true").lower() != "false"
CONFIDENCE_GATE_ENABLED = _env_bool("CONFIDENCE_GATE_ENABLED", False)
LOW_CONF_TOP_MARGIN = float(os.getenv("LOW_CONF_TOP_MARGIN", "0.05"))
LOW_CONF_ROOT_SHARE = float(os.getenv("LOW_CONF_ROOT_SHARE", "0.45"))
LOW_CONF_TOP_SCORE = float(os.getenv("LOW_CONF_TOP_SCORE", "0.20"))
ENABLE_ANCHOR_GATE = os.getenv("ENABLE_ANCHOR_GATE", "true").lower() != "false"
QUERY_PLAN_ENABLED = _env_bool("QUERY_PLAN_ENABLED", False)
RAG_CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "0"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "0"))
RERANK_CPU_TOP_N_CAP = int(os.getenv("RERANK_CPU_TOP_N_CAP", "0"))
RERANK_INPUT_K_CPU = int(os.getenv("RERANK_INPUT_K_CPU", "10"))
RERANK_INPUT_K_GPU = int(os.getenv("RERANK_INPUT_K_GPU", "20"))
RERANK_CACHE_ENABLED = _env_bool("RERANK_CACHE_ENABLED", True)
RERANK_CACHE_TTL_SECONDS = int(os.getenv("RERANK_CACHE_TTL_SECONDS", "60"))
MILVUS_SEARCH_EF = int(os.getenv("MILVUS_SEARCH_EF", "64"))
MILVUS_SPARSE_DROP_RATIO = float(os.getenv("MILVUS_SPARSE_DROP_RATIO", "0.2"))
MILVUS_RRF_K = int(os.getenv("MILVUS_RRF_K", "60"))
DOC_SCOPE_GLOBAL_RESERVE_WEIGHT = min(1.0, max(0.0, float(os.getenv("DOC_SCOPE_GLOBAL_RESERVE_WEIGHT", "0.2"))))
DOC_SCOPE_FILENAME_BOOST_WEIGHT = float(os.getenv("DOC_SCOPE_FILENAME_BOOST_WEIGHT", "0.15"))
RERANK_PAIR_ENRICHMENT_ENABLED = _env_bool("RERANK_PAIR_ENRICHMENT_ENABLED", False)
HEADING_LEXICAL_ENABLED = _env_bool("HEADING_LEXICAL_ENABLED", False)
HEADING_LEXICAL_WEIGHT = min(1.0, max(0.0, float(os.getenv("HEADING_LEXICAL_WEIGHT", "0.20"))))
RAG_INDEX_PROFILE = current_index_profile()
RERANK_SCORE_FUSION_ENABLED = _env_bool("RERANK_SCORE_FUSION_ENABLED", False)
RERANK_FUSION_RERANK_WEIGHT = float(os.getenv("RERANK_FUSION_RERANK_WEIGHT", "0.65"))
RERANK_FUSION_RRF_WEIGHT = float(os.getenv("RERANK_FUSION_RRF_WEIGHT", "0.20"))
RERANK_FUSION_SCOPE_WEIGHT = float(os.getenv("RERANK_FUSION_SCOPE_WEIGHT", "0.10"))
RERANK_FUSION_METADATA_WEIGHT = float(os.getenv("RERANK_FUSION_METADATA_WEIGHT", "0.05"))

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


def _get_parent_chunk_store():
    global _parent_chunk_store
    if _parent_chunk_store is None:
        from parent_chunk_store import ParentChunkStore

        _parent_chunk_store = ParentChunkStore()
    return _parent_chunk_store


def _doc_retrieval_text(doc: dict) -> str:
    return str(doc.get("retrieval_text") or doc.get("text") or "")


def _build_enriched_pair(doc: dict) -> str:
    return _rerank_build_enriched_pair(doc, doc_text_getter=_doc_retrieval_text)


def _weighted_rrf_merge(
    result_sets: list[tuple[list[dict], float]],
    rrf_k: int = 60,
) -> list[dict]:
    return _retrieval_weighted_rrf_merge(result_sets, rrf_k=rrf_k)


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


def _annotate_scope_scores(docs: list[dict], matched_files: list[tuple[str, float]]) -> list[dict]:
    return _retrieval_annotate_scope_scores(docs, matched_files)


def _finish_retrieval_pipeline(
    query: str,
    search_query: str,
    retrieved: list[dict],
    top_k: int,
    candidate_k: int,
    timings: Dict[str, float],
    stage_errors: list[Dict[str, str]],
    total_start: float,
    extra_trace: dict | None = None,
    query_plan: QueryPlan | None = None,
    context_files: list[str] | None = None,
    base_filter: str | None = None,
) -> Dict[str, Any]:
    """Complete the retrieval pipeline: rerank -> structure_rerank -> confidence_gate."""
    current_stage = "rerank"
    try:
        stage_start = time.perf_counter()
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        timings["rerank_ms"] = _elapsed_ms(stage_start)
        if rerank_meta.get("rerank_error"):
            stage_errors.append(_stage_error("rerank", str(rerank_meta.get("rerank_error")), "ranked_candidates"))

        current_stage = "structure_rerank"
        stage_start = time.perf_counter()
        reranked_docs, structure_meta = _apply_structure_rerank(docs=reranked, top_k=top_k)
        timings["structure_rerank_ms"] = _elapsed_ms(stage_start)

        current_stage = "confidence_gate"
        stage_start = time.perf_counter()
        confidence_meta = _evaluate_retrieval_confidence(query=query, docs=reranked_docs)
        timings["confidence_ms"] = _elapsed_ms(stage_start)
        timings["total_retrieve_ms"] = _elapsed_ms(total_start)

        rerank_meta["retrieval_mode"] = "hybrid_scoped" if (extra_trace and extra_trace.get("scope_filter_applied")) else "hybrid"
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
        rerank_meta["hybrid_error"] = None
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

        return {"docs": reranked_docs, "meta": rerank_meta}

    except Exception as exc:
        stage_errors.append(_stage_error(current_stage, str(exc)))
        timings["total_retrieve_ms"] = _elapsed_ms(total_start)
        return {
            "docs": [],
            "meta": {
                "rerank_enabled": _is_rerank_enabled(),
                "rerank_applied": False,
                "rerank_model": RERANK_MODEL,
                "rerank_endpoint": _get_rerank_endpoint(),
                "rerank_error": str(exc),
                "hybrid_error": None,
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
            },
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


def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


elapsed_ms = _elapsed_ms


def _effective_candidate_k(top_k: int) -> int:
    if RAG_CANDIDATE_K > 0:
        return max(top_k, RAG_CANDIDATE_K)
    return max(top_k * 10, 50)


def _effective_rerank_top_n(top_k: int, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    requested = RERANK_TOP_N if RERANK_TOP_N > 0 else top_k
    requested = max(top_k, requested)
    if _rerank_device_tier() == "cpu" and RERANK_CPU_TOP_N_CAP > 0:
        requested = min(requested, RERANK_CPU_TOP_N_CAP)
    return min(candidate_count, requested)


def _rerank_device_tier() -> str:
    device = (RERANK_DEVICE or "auto").lower()
    if device in ("cuda", "gpu", "mps"):
        return "gpu"
    if device == "auto":
        try:
            import torch

            return "gpu" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


def _effective_rerank_input_k(rerank_top_n: int, candidate_count: int) -> tuple[int, str, int]:
    if candidate_count <= 0:
        return 0, _rerank_device_tier(), 0
    device_tier = _rerank_device_tier()
    cap = RERANK_INPUT_K_GPU if device_tier == "gpu" else RERANK_INPUT_K_CPU
    if cap <= 0:
        return candidate_count, device_tier, cap
    return min(candidate_count, cap), device_tier, cap


def _is_rerank_enabled() -> bool:
    is_local = RERANK_PROVIDER == "local" and bool(RERANK_MODEL)
    is_ollama = RERANK_PROVIDER == "ollama" and bool(RERANK_MODEL and RERANK_BINDING_HOST)
    is_api = RERANK_PROVIDER not in ("local", "ollama") and bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST)
    return is_local or is_ollama or is_api


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
        "version": 1,
        "query": query,
        "provider": RERANK_PROVIDER,
        "model": RERANK_MODEL or "",
        "binding_host": RERANK_BINDING_HOST or "",
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
        from sentence_transformers import CrossEncoder

        device = RERANK_DEVICE
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        _local_reranker = CrossEncoder(RERANK_MODEL, device=device)
    return _local_reranker


_RETRIEVE_TIMING_KEYS = (
    "embed_dense_ms",
    "embed_sparse_ms",
    "milvus_hybrid_ms",
    "milvus_dense_fallback_ms",
    "rerank_ms",
    "structure_rerank_ms",
    "confidence_ms",
    "total_retrieve_ms",
)


def _ensure_retrieve_timing_defaults(timings: Dict[str, float]) -> Dict[str, float]:
    for key in _RETRIEVE_TIMING_KEYS:
        timings.setdefault(key, 0.0)
    return timings


def _stage_error(stage: str, error: str, fallback_to: str | None = None) -> Dict[str, str]:
    return StageError(stage=stage, error=error, fallback_to=fallback_to).as_dict()


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


def _apply_structure_rerank(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    return _context_apply_structure_rerank(
        docs,
        top_k,
        enabled=STRUCTURE_RERANK_ENABLED,
        root_weight=STRUCTURE_RERANK_ROOT_WEIGHT,
        same_root_cap=SAME_ROOT_CAP,
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
        binding_host=RERANK_BINDING_HOST,
        api_key=RERANK_API_KEY,
        cpu_top_n_cap=RERANK_CPU_TOP_N_CAP,
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
        get_endpoint=_get_rerank_endpoint,
        effective_top_n=_effective_rerank_top_n,
        effective_input_k=_effective_rerank_input_k,
        get_local_reranker=_get_local_reranker,
        cache_key=_rerank_cache_key,
        load_cached_result=_load_cached_rerank_result,
        store_result=_store_rerank_result,
        doc_text_getter=_doc_retrieval_text,
        post=requests.post,
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


def _dedupe_docs(docs: list[dict]) -> list[dict]:
    return _retrieval_dedupe_docs(docs)


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

    docs = _dedupe_docs(docs)
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
    total_start = time.perf_counter()
    timings: Dict[str, float] = {}
    stage_errors: list[Dict[str, str]] = []
    current_stage = "query_plan"
    candidate_k = _effective_candidate_k(top_k)

    # --- QueryPlan parsing ---
    stage_start = time.perf_counter()
    if QUERY_PLAN_ENABLED:
        filename_registry = get_filename_registry(_milvus_manager, cache)
        query_plan = parse_query_plan(
            raw_query=query,
            filename_registry=filename_registry,
            context_files=context_files,
        )
    else:
        query_plan = QueryPlan(
            raw_query=query,
            semantic_query=query,
            clean_query=query,
            doc_hints=[],
            matched_files=[],
            scope_mode="none",
            heading_hint=None,
            anchors=[],
            model_numbers=[],
            intent_type=None,
            route="global_hybrid",
        )
    timings["query_plan_ms"] = _elapsed_ms(stage_start)

    # Determine the search query (semantic_query, not raw_query)
    search_query = query_plan.semantic_query

    # Build base filter
    base_filter = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"

    # Context files override (highest priority)
    filename_filter = _build_filename_filter(context_files)
    if filename_filter:
        base_filter = f"{base_filter} and {filename_filter}"

    # --- Scoped + Global parallel retrieval ---
    routable_matched_files = [
        (filename, score)
        for filename, score in query_plan.matched_files
        if score >= DOC_SCOPE_MATCH_BOOST
    ]

    if QUERY_PLAN_ENABLED and query_plan.scope_mode == "filter" and routable_matched_files and not filename_filter:
        # Build scoped filter
        matched_filenames = [f for f, _ in routable_matched_files]
        scoped_filename_filter = _build_filename_filter(matched_filenames)
        scoped_filter = f"{base_filter} and {scoped_filename_filter}"

        # Compute embeddings once, reuse for both paths
        current_stage = "embed_dense"
        try:
            stage_start = time.perf_counter()
            dense_embeddings = _embedding_service.get_embeddings([search_query])
            dense_embedding = dense_embeddings[0]
            timings["embed_dense_ms"] = _elapsed_ms(stage_start)

            current_stage = "embed_sparse"
            stage_start = time.perf_counter()
            sparse_embedding = _embedding_service.get_sparse_embedding(search_query)
            timings["embed_sparse_ms"] = _elapsed_ms(stage_start)
        except Exception as exc:
            stage_errors.append(_stage_error(current_stage, str(exc), "dense_retrieve"))
            # Fall through to error handling below
            dense_embedding = None
            sparse_embedding = None

        if dense_embedding is not None and sparse_embedding is not None:
            # Parallel scoped + global retrieval via ThreadPoolExecutor
            current_stage = "scoped_global_retrieve"
            stage_start = time.perf_counter()

            def _scoped_retrieve():
                return _milvus_manager.hybrid_retrieve(
                    dense_embedding=dense_embedding,
                    sparse_embedding=sparse_embedding,
                    top_k=candidate_k,
                    rrf_k=MILVUS_RRF_K,
                    search_ef=MILVUS_SEARCH_EF,
                    sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
                    filter_expr=scoped_filter,
                )

            def _global_retrieve():
                return _milvus_manager.hybrid_retrieve(
                    dense_embedding=dense_embedding,
                    sparse_embedding=sparse_embedding,
                    top_k=candidate_k,
                    rrf_k=MILVUS_RRF_K,
                    search_ef=MILVUS_SEARCH_EF,
                    sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
                    filter_expr=base_filter,
                )

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

            timings["milvus_hybrid_ms"] = _elapsed_ms(stage_start)
            scoped = _annotate_scope_scores(scoped, routable_matched_files)
            global_ = _annotate_scope_scores(global_, routable_matched_files)

            # Weighted RRF merge: scoped 80%, global 20%
            retrieved = _weighted_rrf_merge(
                [(scoped, 1.0 - DOC_SCOPE_GLOBAL_RESERVE_WEIGHT), (global_, DOC_SCOPE_GLOBAL_RESERVE_WEIGHT)],
                rrf_k=MILVUS_RRF_K,
            )

            scope_trace = {
                "query_plan": query_plan.to_dict(),
                "query_plan_enabled": QUERY_PLAN_ENABLED,
                "semantic_query": query_plan.semantic_query,
                "index_profile": RAG_INDEX_PROFILE,
                "v3_layers": ["query_plan", "doc_resolver", "scoped_hybrid", "weighted_rrf", "rerank", "structure_rerank"],
                "scope_mode": query_plan.scope_mode,
                "query_route": query_plan.route,
                "scoped_candidate_count": len(scoped),
                "global_candidate_count": len(global_),
                "scope_filter_applied": True,
                "filename_boost_applied": False,
                "matched_files_top3": [(f, round(s, 3)) for f, s in query_plan.matched_files[:3]],
                "doc_scope_match_ratios": [round(s, 3) for _, s in query_plan.matched_files[:3]],
            }
        else:
            # Embedding failed, fall through to error handling
            retrieved = []
            scope_trace = {
                "query_plan": query_plan.to_dict(),
                "query_plan_enabled": QUERY_PLAN_ENABLED,
                "semantic_query": query_plan.semantic_query,
                "index_profile": RAG_INDEX_PROFILE,
                "v3_layers": ["query_plan", "doc_resolver", "scoped_hybrid", "embedding_failed"],
                "scope_mode": query_plan.scope_mode,
                "query_route": query_plan.route,
                "scope_filter_applied": False,
                "filename_boost_applied": False,
                "embedding_failed": True,
            }

        # Apply heading lexical scoring if enabled
        if QUERY_PLAN_ENABLED and HEADING_LEXICAL_ENABLED and query_plan.scope_mode in {"filter", "boost"} and query_plan.heading_hint:
            retrieved = _apply_heading_lexical_scoring(
                query_plan=query_plan,
                candidates=retrieved,
            )

        # Continue with rerank + structure_rerank + confidence
        return _finish_retrieval_pipeline(
            query=query,
            search_query=search_query,
            retrieved=retrieved,
            top_k=top_k,
            candidate_k=candidate_k,
            timings=timings,
            stage_errors=stage_errors,
            total_start=total_start,
            extra_trace=scope_trace,
            query_plan=query_plan,
            context_files=context_files,
            base_filter=base_filter,
        )

    # --- Standard (non-scoped) retrieval path ---
    filter_expr = base_filter
    if filename_filter:
        filter_expr = f"{filter_expr} and {filename_filter}"
    hybrid_error = None
    global_trace = {
        "query_plan": query_plan.to_dict(),
        "query_plan_enabled": QUERY_PLAN_ENABLED,
        "semantic_query": query_plan.semantic_query,
        "index_profile": RAG_INDEX_PROFILE,
        "v3_layers": ["query_plan", "global_hybrid", "rerank", "structure_rerank"],
        "scope_mode": query_plan.scope_mode,
        "query_route": query_plan.route,
        "scope_filter_applied": False,
        "filename_boost_applied": False,
        "filename_boosted_candidate_count": 0,
        "matched_files_top3": [(f, round(s, 3)) for f, s in query_plan.matched_files[:3]],
        "doc_scope_match_ratios": [round(s, 3) for _, s in query_plan.matched_files[:3]],
    }
    try:
        stage_start = time.perf_counter()
        dense_embeddings = _embedding_service.get_embeddings([search_query])
        dense_embedding = dense_embeddings[0]
        timings["embed_dense_ms"] = _elapsed_ms(stage_start)

        current_stage = "embed_sparse"
        stage_start = time.perf_counter()
        sparse_embedding = _embedding_service.get_sparse_embedding(search_query)
        timings["embed_sparse_ms"] = _elapsed_ms(stage_start)

        current_stage = "hybrid_retrieve"
        stage_start = time.perf_counter()
        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            rrf_k=MILVUS_RRF_K,
            search_ef=MILVUS_SEARCH_EF,
            sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
            filter_expr=filter_expr,
        )
        timings["milvus_hybrid_ms"] = _elapsed_ms(stage_start)

        if QUERY_PLAN_ENABLED and query_plan.scope_mode == "boost":
            retrieved = _apply_filename_boost(query_plan, retrieved)
            boosted_count = sum(1 for doc in retrieved if doc.get("filename_boost_applied"))
            global_trace["filename_boost_applied"] = boosted_count > 0
            global_trace["filename_boosted_candidate_count"] = boosted_count

        current_stage = "rerank"
        stage_start = time.perf_counter()
        reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
        timings["rerank_ms"] = _elapsed_ms(stage_start)
        if rerank_meta.get("rerank_error"):
            stage_errors.append(_stage_error("rerank", str(rerank_meta.get("rerank_error")), "ranked_candidates"))

        current_stage = "structure_rerank"
        stage_start = time.perf_counter()
        reranked_docs, structure_meta = _apply_structure_rerank(docs=reranked, top_k=top_k)
        timings["structure_rerank_ms"] = _elapsed_ms(stage_start)

        current_stage = "confidence_gate"
        stage_start = time.perf_counter()
        confidence_meta = _evaluate_retrieval_confidence(query=query, docs=reranked_docs)
        timings["confidence_ms"] = _elapsed_ms(stage_start)
        timings["total_retrieve_ms"] = _elapsed_ms(total_start)
        rerank_meta["retrieval_mode"] = "hybrid_boosted" if global_trace.get("filename_boost_applied") else "hybrid"
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
        rerank_meta["hybrid_error"] = None
        rerank_meta["dense_error"] = None
        rerank_meta["timings"] = dict(_ensure_retrieve_timing_defaults(timings))
        rerank_meta["stage_errors"] = stage_errors
        rerank_meta.update(structure_meta)
        rerank_meta.update(confidence_meta)
        rerank_meta["candidates_before_rerank"] = _candidate_trace(retrieved)
        rerank_meta["candidates_after_rerank"] = _candidate_trace(reranked)
        rerank_meta["candidates_after_structure_rerank"] = _candidate_trace(reranked_docs)
        rerank_meta.update(global_trace)
        return {"docs": reranked_docs, "meta": rerank_meta}
    except Exception as exc:
        hybrid_error = str(exc)
        stage_errors.append(_stage_error(current_stage, hybrid_error, "dense_retrieve"))
        try:
            current_stage = "embed_dense_fallback"
            stage_start = time.perf_counter()
            dense_embeddings = _embedding_service.get_embeddings([search_query])
            dense_embedding = dense_embeddings[0]
            fallback_dense_ms = _elapsed_ms(stage_start)
            if "embed_dense_ms" in timings:
                timings["embed_dense_fallback_ms"] = fallback_dense_ms
            else:
                timings["embed_dense_ms"] = fallback_dense_ms

            current_stage = "dense_retrieve"
            stage_start = time.perf_counter()
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=candidate_k,
                search_ef=MILVUS_SEARCH_EF,
                filter_expr=filter_expr,
            )
            timings["milvus_dense_fallback_ms"] = _elapsed_ms(stage_start)

            if QUERY_PLAN_ENABLED and query_plan.scope_mode == "boost":
                retrieved = _apply_filename_boost(query_plan, retrieved)
                boosted_count = sum(1 for doc in retrieved if doc.get("filename_boost_applied"))
                global_trace["filename_boost_applied"] = boosted_count > 0
                global_trace["filename_boosted_candidate_count"] = boosted_count

            current_stage = "rerank"
            stage_start = time.perf_counter()
            reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            timings["rerank_ms"] = _elapsed_ms(stage_start)
            if rerank_meta.get("rerank_error"):
                stage_errors.append(_stage_error("rerank", str(rerank_meta.get("rerank_error")), "ranked_candidates"))

            current_stage = "structure_rerank"
            stage_start = time.perf_counter()
            reranked_docs, structure_meta = _apply_structure_rerank(docs=reranked, top_k=top_k)
            timings["structure_rerank_ms"] = _elapsed_ms(stage_start)

            current_stage = "confidence_gate"
            stage_start = time.perf_counter()
            confidence_meta = _evaluate_retrieval_confidence(query=query, docs=reranked_docs)
            timings["confidence_ms"] = _elapsed_ms(stage_start)
            timings["total_retrieve_ms"] = _elapsed_ms(total_start)
            rerank_meta["retrieval_mode"] = (
                "dense_fallback_boosted" if global_trace.get("filename_boost_applied") else "dense_fallback"
            )
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
            rerank_meta.update(global_trace)
            return {"docs": reranked_docs, "meta": rerank_meta}
        except Exception as dense_exc:
            dense_error = str(dense_exc)
            stage_errors.append(_stage_error(current_stage, dense_error))
            timings["total_retrieve_ms"] = _elapsed_ms(total_start)
            return {
                "docs": [],
                "meta": {
                    "rerank_enabled": _is_rerank_enabled(),
                    "rerank_applied": False,
                    "rerank_model": RERANK_MODEL,
                    "rerank_endpoint": _get_rerank_endpoint(),
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
                    **global_trace,
                },
            }
