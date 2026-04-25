from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any
import os
import json
import hashlib
import re
import time
import requests
from difflib import SequenceMatcher
from dotenv import load_dotenv

from cache import cache
from milvus_client import MilvusManager
from embedding import embedding_service as _embedding_service
from parent_chunk_store import ParentChunkStore
from langchain.chat_models import init_chat_model
from query_plan import (
    QueryPlan,
    DOC_SCOPE_MATCH_BOOST,
    parse_query_plan,
    get_filename_registry,
    _normalize_filename as _qp_normalize_filename,
    _filename_match_score as _qp_filename_match_score,
)

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
RAG_CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "0"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "0"))
RERANK_CPU_TOP_N_CAP = int(os.getenv("RERANK_CPU_TOP_N_CAP", "0"))
RERANK_INPUT_K_CPU = int(os.getenv("RERANK_INPUT_K_CPU", "10"))
RERANK_INPUT_K_GPU = int(os.getenv("RERANK_INPUT_K_GPU", "20"))
RERANK_CACHE_ENABLED = os.getenv("RERANK_CACHE_ENABLED", "true").lower() != "false"
RERANK_CACHE_TTL_SECONDS = int(os.getenv("RERANK_CACHE_TTL_SECONDS", "60"))
MILVUS_SEARCH_EF = int(os.getenv("MILVUS_SEARCH_EF", "64"))
MILVUS_SPARSE_DROP_RATIO = float(os.getenv("MILVUS_SPARSE_DROP_RATIO", "0.2"))
MILVUS_RRF_K = int(os.getenv("MILVUS_RRF_K", "60"))
DOC_SCOPE_GLOBAL_RESERVE_WEIGHT = float(os.getenv("DOC_SCOPE_GLOBAL_RESERVE_WEIGHT", "0.2"))
DOC_SCOPE_FILENAME_BOOST_WEIGHT = float(os.getenv("DOC_SCOPE_FILENAME_BOOST_WEIGHT", "0.15"))
RERANK_PAIR_ENRICHMENT_ENABLED = os.getenv("RERANK_PAIR_ENRICHMENT_ENABLED", "false").lower() != "false"
HEADING_LEXICAL_ENABLED = os.getenv("HEADING_LEXICAL_ENABLED", "false").lower() != "false"
HEADING_LEXICAL_WEIGHT = float(os.getenv("HEADING_LEXICAL_WEIGHT", "0.20"))

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()

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


def _doc_retrieval_text(doc: dict) -> str:
    return str(doc.get("retrieval_text") or doc.get("text") or "")


def _build_enriched_pair(doc: dict) -> str:
    """Build enriched pair text for reranker from document metadata."""
    filename = str(doc.get("filename") or "")
    section_path = str(doc.get("section_path") or doc.get("section_title") or "")
    page = doc.get("page_number") or doc.get("page_start") or ""
    anchor = str(doc.get("anchor_id") or doc.get("anchor") or "")
    heading = str(doc.get("section_title") or "")
    body = _doc_retrieval_text(doc)

    prefix_parts = []
    if filename:
        prefix_parts.append(f"[{filename}]")
    if section_path:
        prefix_parts.append(f"[{section_path}]")
    if page:
        prefix_parts.append(f"[p.{page}]")
    if anchor:
        prefix_parts.append(f"[{anchor}]")
    prefix = "".join(prefix_parts)

    if heading and heading in body:
        pair_text = f"{prefix} {body}"
    else:
        pair_text = f"{prefix} {heading}\n{body}" if heading else f"{prefix} {body}"

    return pair_text.strip()


def _weighted_rrf_merge(
    result_sets: list[tuple[list[dict], float]],
    rrf_k: int = 60,
) -> list[dict]:
    """Merge multiple retrieval result sets using weighted RRF.

    Args:
        result_sets: List of (docs, weight) tuples. Weights should sum to ~1.0.
        rrf_k: RRF constant (default 60).

    Returns:
        Merged and deduplicated list of documents sorted by weighted RRF score.
    """
    scores: Dict[str, float] = defaultdict(float)
    doc_by_id: Dict[str, dict] = {}

    for docs, weight in result_sets:
        for rank_idx, doc in enumerate(docs, 1):
            chunk_id = str(doc.get("chunk_id") or doc.get("id") or "")
            if not chunk_id:
                continue
            rrf_score = weight / (rrf_k + rank_idx)
            scores[chunk_id] += rrf_score
            if chunk_id not in doc_by_id:
                doc_by_id[chunk_id] = doc
            else:
                # Merge: keep existing doc but update score
                pass

    sorted_ids = sorted(scores, key=lambda x: -scores[x])
    result = []
    for cid in sorted_ids:
        doc = dict(doc_by_id[cid])
        doc["rrf_merged_score"] = round(scores[cid], 6)
        result.append(doc)

    return result


def _doc_filename(doc: dict) -> str:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    return str(doc.get("filename") or metadata.get("filename") or "")


def _apply_filename_boost(query_plan: QueryPlan, candidates: list[dict]) -> list[dict]:
    """Soft-rank candidates from matched files in boost mode without hard filtering."""
    if query_plan.scope_mode != "boost" or not candidates:
        return candidates

    matched_scores = {
        str(filename): score
        for filename, score in query_plan.matched_files
        if score >= DOC_SCOPE_MATCH_BOOST
    }
    if not matched_scores:
        return candidates

    normalized_scores = {
        _qp_normalize_filename(filename): score
        for filename, score in matched_scores.items()
    }

    scored: list[tuple[float, dict]] = []
    for idx, doc in enumerate(candidates, 1):
        filename = _doc_filename(doc)
        match_score = matched_scores.get(filename, 0.0)
        if not match_score:
            match_score = normalized_scores.get(_qp_normalize_filename(filename), 0.0)

        rank_score = 1.0 / (MILVUS_RRF_K + idx)
        boosted_score = rank_score + (DOC_SCOPE_FILENAME_BOOST_WEIGHT * match_score)
        next_doc = dict(doc)
        if match_score:
            next_doc["filename_boost_applied"] = True
            next_doc["filename_boost_match_score"] = round(match_score, 6)
            next_doc["filename_boost_score"] = round(boosted_score, 6)
        scored.append((boosted_score, next_doc))

    scored.sort(key=lambda item: -item[0])
    return [doc for _, doc in scored]


def _apply_heading_lexical_scoring(
    query_plan: QueryPlan,
    candidates: list[dict],
) -> list[dict]:
    """Apply heading/section lexical scoring to candidates.

    Only active when scope_mode in {filter, boost} and heading_hint exists.
    """
    if not query_plan.heading_hint or query_plan.scope_mode not in {"filter", "boost"}:
        return candidates

    alpha = HEADING_LEXICAL_WEIGHT
    semantic_query = query_plan.semantic_query
    anchors = query_plan.anchors

    scored_candidates = []
    for idx, doc in enumerate(candidates, 1):
        section_path = str(doc.get("section_path") or "")
        heading = str(doc.get("section_title") or "")

        heading_lexical_score = (
            0.5 * SequenceMatcher(None, semantic_query, section_path).ratio()
            + 0.3 * SequenceMatcher(None, semantic_query, heading).ratio()
            + 0.2 * (1.0 if any(a in heading or a in section_path for a in anchors) else 0.0)
        )

        # Normalize RRF-like rank score: 1/(k+rank)
        rrf_rank_normalized = 1.0 / (MILVUS_RRF_K + idx)

        final_sort_key = (1 - alpha) * rrf_rank_normalized + alpha * heading_lexical_score
        scored_candidates.append((final_sort_key, doc))

    scored_candidates.sort(key=lambda x: -x[0])
    return [doc for _, doc in scored_candidates]


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
                "root_chunk_id": doc.get("root_chunk_id", ""),
                "anchor_id": doc.get("anchor_id", ""),
                "filename": doc.get("filename", ""),
                "text_preview": _doc_retrieval_text(doc)[:240],
                "score": doc.get("score"),
                "rerank_score": doc.get("rerank_score"),
                "final_score": doc.get("final_score"),
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


def _effective_candidate_k(top_k: int) -> int:
    if RAG_CANDIDATE_K > 0:
        return max(top_k, RAG_CANDIDATE_K)
    return max(top_k * 10, 50)


def _effective_rerank_top_n(top_k: int, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    requested = RERANK_TOP_N if RERANK_TOP_N > 0 else top_k
    requested = max(top_k, requested)
    if RERANK_DEVICE.lower() == "cpu" and RERANK_CPU_TOP_N_CAP > 0:
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
    return min(candidate_count, max(rerank_top_n, cap)), device_tier, cap


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
        reranked.append(doc)

    return reranked if reranked else None


def _store_rerank_result(cache_key: str, reranked: List[dict], docs_for_rerank: List[dict]) -> None:
    rank_to_index = {doc.get("rrf_rank"): idx for idx, doc in enumerate(docs_for_rerank)}
    items = []
    for doc in reranked:
        index = rank_to_index.get(doc.get("rrf_rank"))
        if index is None:
            return
        items.append({"index": index, "rerank_score": doc.get("rerank_score")})
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
    item = {"stage": stage, "error": error}
    if fallback_to:
        item["fallback_to"] = fallback_to
    return item


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    deduped: List[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    if not AUTO_MERGE_ENABLED or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    # The current loader stores L1 parent chunks and L3 leaves; no L2 rows exist.
    merged_docs, merged_count = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    merged_docs = merged_docs[:top_k]

    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": merged_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": merged_count,
        "auto_merge_steps": int(merged_count > 0),
        "auto_merge_path": "L3->L1",
    }


def _apply_structure_rerank(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    if not STRUCTURE_RERANK_ENABLED:
        limited = docs[:top_k]
        return limited, {
            "structure_rerank_enabled": False,
            "structure_rerank_applied": False,
            "structure_rerank_root_weight": STRUCTURE_RERANK_ROOT_WEIGHT,
            "same_root_cap": SAME_ROOT_CAP,
            "dominant_root_id": None,
            "dominant_root_share": 0.0,
            "dominant_root_support": 0,
        }
    if not docs:
        return [], {
            "structure_rerank_enabled": STRUCTURE_RERANK_ENABLED,
            "structure_rerank_applied": False,
            "structure_rerank_root_weight": STRUCTURE_RERANK_ROOT_WEIGHT,
            "same_root_cap": SAME_ROOT_CAP,
            "dominant_root_id": None,
            "dominant_root_share": 0.0,
            "dominant_root_support": 0,
        }

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        if not root_id:
            root_id = f"__orphan__:{len(grouped)}"
        grouped[root_id].append(doc)

    root_scores: Dict[str, float] = {}
    for root_id, items in grouped.items():
        root_scores[root_id] = max(
            float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
            for item in items
        )

    scored_docs = []
    for doc in docs:
        leaf_score = float(doc.get("rerank_score", doc.get("score", 0.0)) or 0.0)
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        root_score = root_scores.get(root_id, leaf_score)
        final_score = (1.0 - STRUCTURE_RERANK_ROOT_WEIGHT) * leaf_score + STRUCTURE_RERANK_ROOT_WEIGHT * root_score
        enriched = dict(doc)
        enriched["leaf_score"] = leaf_score
        enriched["root_score"] = root_score
        enriched["final_score"] = final_score
        scored_docs.append(enriched)

    scored_docs.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
    limited: List[dict] = []
    per_root: Dict[str, int] = defaultdict(int)
    for doc in scored_docs:
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        if per_root[root_id] >= SAME_ROOT_CAP:
            continue
        limited.append(doc)
        per_root[root_id] += 1
        if len(limited) >= top_k:
            break

    root_total_scores: Dict[str, float] = defaultdict(float)
    for doc in limited:
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        root_total_scores[root_id] += float(doc.get("final_score", 0.0) or 0.0)

    dominant_root_id = None
    dominant_root_share = 0.0
    dominant_root_support = 0
    total_score = sum(root_total_scores.values())
    if root_total_scores:
        dominant_root_id = max(root_total_scores, key=root_total_scores.get)
        dominant_root_support = per_root.get(dominant_root_id, 0)
        if total_score > 0:
            dominant_root_share = root_total_scores[dominant_root_id] / total_score

    return limited, {
        "structure_rerank_enabled": STRUCTURE_RERANK_ENABLED,
        "structure_rerank_applied": True,
        "structure_rerank_root_weight": STRUCTURE_RERANK_ROOT_WEIGHT,
        "same_root_cap": SAME_ROOT_CAP,
        "dominant_root_id": dominant_root_id,
        "dominant_root_share": dominant_root_share,
        "dominant_root_support": dominant_root_support,
    }


def _extract_query_anchors(query: str) -> list[str]:
    if not query:
        return []
    return list(dict.fromkeys(_ANCHOR_PATTERN.findall(query)))


def _doc_matches_anchor(doc: dict, anchor: str) -> bool:
    if not anchor:
        return False
    for value in (
        doc.get("anchor_id"),
        doc.get("section_title"),
        doc.get("section_path"),
    ):
        if anchor and anchor in str(value or ""):
            return True
    prefix = _doc_retrieval_text(doc).split("\n", 1)[0]
    return anchor in prefix


def _evaluate_retrieval_confidence(query: str, docs: List[dict]) -> Dict[str, Any]:
    if not docs:
        return {
            "confidence_gate_enabled": CONFIDENCE_GATE_ENABLED,
            "fallback_required": CONFIDENCE_GATE_ENABLED,
            "confidence_reasons": ["no_docs"] if CONFIDENCE_GATE_ENABLED else [],
            "top_margin": 0.0,
            "top_score": 0.0,
            "dominant_root_share": 0.0,
            "dominant_root_support": 0,
            "anchor_match": False,
            "query_anchors": [],
        }

    top_score = float(docs[0].get("final_score", docs[0].get("rerank_score", docs[0].get("score", 0.0))) or 0.0)
    second_score = (
        float(docs[1].get("final_score", docs[1].get("rerank_score", docs[1].get("score", 0.0))) or 0.0)
        if len(docs) > 1
        else 0.0
    )
    top_margin = top_score - second_score

    root_total_scores: Dict[str, float] = defaultdict(float)
    root_supports: Dict[str, int] = defaultdict(int)
    for doc in docs[:5]:
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        score = float(doc.get("final_score", doc.get("rerank_score", doc.get("score", 0.0))) or 0.0)
        root_total_scores[root_id] += score
        root_supports[root_id] += 1

    dominant_root_share = 0.0
    dominant_root_support = 0
    if root_total_scores:
        dominant_root_id = max(root_total_scores, key=root_total_scores.get)
        total = sum(root_total_scores.values())
        dominant_root_support = root_supports[dominant_root_id]
        dominant_root_share = (root_total_scores[dominant_root_id] / total) if total else 0.0

    anchors = _extract_query_anchors(query) if ENABLE_ANCHOR_GATE else []
    anchor_match = True
    if anchors:
        anchor_match = False
        for anchor in anchors:
            if any(_doc_matches_anchor(doc, anchor) for doc in docs[:2]):
                anchor_match = True
                break

    reasons: list[str] = []
    if anchors and not anchor_match:
        reasons.append("anchor_mismatch")
    if top_margin < LOW_CONF_TOP_MARGIN and dominant_root_share < LOW_CONF_ROOT_SHARE:
        reasons.append("weak_margin_and_root")
    if top_score < LOW_CONF_TOP_SCORE and top_margin < LOW_CONF_TOP_MARGIN:
        reasons.append("low_score_and_margin")

    if not CONFIDENCE_GATE_ENABLED:
        reasons = []

    return {
        "confidence_gate_enabled": CONFIDENCE_GATE_ENABLED,
        "fallback_required": bool(reasons),
        "confidence_reasons": reasons,
        "top_margin": top_margin,
        "top_score": top_score,
        "dominant_root_share": dominant_root_share,
        "dominant_root_support": dominant_root_support,
        "anchor_match": anchor_match,
        "query_anchors": anchors,
    }


def _rerank_pair_text(doc: dict, enrichment_enabled: bool) -> str:
    """Get the text to send to the reranker, optionally enriched with metadata."""
    if enrichment_enabled:
        return _build_enriched_pair(doc)
    return _doc_retrieval_text(doc)


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    pair_enrichment_enabled = bool(RERANK_PAIR_ENRICHMENT_ENABLED)
    rerank_top_n = _effective_rerank_top_n(top_k, len(docs_with_rank))
    rerank_input_k, rerank_input_device_tier, rerank_input_cap = _effective_rerank_input_k(
        rerank_top_n,
        len(docs_with_rank),
    )
    rerank_top_n = min(rerank_top_n, rerank_input_k)
    docs_for_rerank = docs_with_rank[:rerank_input_k]
    is_local = RERANK_PROVIDER == "local" and bool(RERANK_MODEL)
    is_ollama = RERANK_PROVIDER == "ollama" and bool(RERANK_MODEL and RERANK_BINDING_HOST)
    is_api = RERANK_PROVIDER not in ("local", "ollama") and bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST)
    meta: Dict[str, Any] = {
        "rerank_enabled": is_local or is_ollama or is_api,
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_provider": RERANK_PROVIDER,
        "rerank_endpoint": _get_rerank_endpoint() if is_api else (f"{RERANK_BINDING_HOST.rstrip('/')}/api/rerank" if is_ollama else None),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
        "rerank_top_n": rerank_top_n,
        "rerank_cpu_top_n_cap": RERANK_CPU_TOP_N_CAP,
        "rerank_input_count": rerank_input_k if (is_local or is_ollama or is_api) else 0,
        "rerank_output_count": 0,
        "rerank_input_cap": rerank_input_cap,
        "rerank_input_device_tier": rerank_input_device_tier,
        "rerank_cache_enabled": RERANK_CACHE_ENABLED,
        "rerank_cache_hit": False,
        "rerank_pair_enrichment_enabled": pair_enrichment_enabled,
    }
    if not docs_with_rank or not meta["rerank_enabled"]:
        result = docs_with_rank[:rerank_top_n]
        meta["rerank_output_count"] = len(result)
        return result, meta

    rerank_cache_key = ""
    if RERANK_CACHE_ENABLED and docs_for_rerank and rerank_top_n > 0:
        rerank_cache_key = _rerank_cache_key(
            query,
            docs_for_rerank,
            rerank_top_n,
            rerank_input_k,
            pair_enrichment_enabled,
        )
        cached_result = _load_cached_rerank_result(rerank_cache_key, docs_for_rerank, rerank_top_n)
        if cached_result:
            meta["rerank_applied"] = True
            meta["rerank_cache_hit"] = True
            meta["rerank_output_count"] = len(cached_result)
            return cached_result, meta

    if is_local:
        try:
            reranker = _get_local_reranker()
            if not reranker:
                meta["rerank_error"] = "local_reranker_not_loaded"
                result = docs_for_rerank[:rerank_top_n]
                meta["rerank_output_count"] = len(result)
                return result, meta
            texts = [_rerank_pair_text(doc, pair_enrichment_enabled) for doc in docs_for_rerank]
            pairs = [[query, text] for text in texts]
            scores = reranker.predict(pairs)
            indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            reranked = []
            for idx, score in indexed_scores[:rerank_top_n]:
                doc = dict(docs_for_rerank[idx])
                doc["rerank_score"] = float(score)
                reranked.append(doc)
            meta["rerank_applied"] = True
            result = reranked[:rerank_top_n]
            meta["rerank_output_count"] = len(result)
            if rerank_cache_key:
                _store_rerank_result(rerank_cache_key, result, docs_for_rerank)
            return result, meta
        except Exception as e:
            meta["rerank_error"] = str(e)
            result = docs_for_rerank[:rerank_top_n]
            meta["rerank_output_count"] = len(result)
            return result, meta

    if is_ollama:
        try:
            meta["rerank_applied"] = True
            payload = {
                "model": RERANK_MODEL,
                "query": query,
                "documents": [_rerank_pair_text(doc, pair_enrichment_enabled) for doc in docs_for_rerank],
                "top_n": rerank_top_n,
            }
            response = requests.post(
                meta["rerank_endpoint"],
                json=payload,
                timeout=30,
            )
            if response.status_code >= 400:
                meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
                result = docs_for_rerank[:rerank_top_n]
                meta["rerank_output_count"] = len(result)
                return result, meta

            items = response.json().get("results", [])
            reranked = []
            for item in items:
                idx = item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(docs_for_rerank):
                    doc = dict(docs_for_rerank[idx])
                    score = item.get("relevance_score")
                    if score is not None:
                        doc["rerank_score"] = score
                    reranked.append(doc)

            if reranked:
                result = reranked[:rerank_top_n]
                meta["rerank_output_count"] = len(result)
                if rerank_cache_key:
                    _store_rerank_result(rerank_cache_key, result, docs_for_rerank)
                return result, meta

            meta["rerank_error"] = "empty_rerank_results"
            result = docs_for_rerank[:rerank_top_n]
            meta["rerank_output_count"] = len(result)
            return result, meta
        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            meta["rerank_error"] = str(e)
            result = docs_for_rerank[:rerank_top_n]
            meta["rerank_output_count"] = len(result)
            return result, meta

    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": [_rerank_pair_text(doc, pair_enrichment_enabled) for doc in docs_for_rerank],
        "top_n": rerank_top_n,
        "return_documents": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RERANK_API_KEY}",
    }
    try:
        meta["rerank_applied"] = True
        response = requests.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=15,
        )
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
            result = docs_for_rerank[:rerank_top_n]
            meta["rerank_output_count"] = len(result)
            return result, meta

        items = response.json().get("results", [])
        reranked = []
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_for_rerank):
                doc = dict(docs_for_rerank[idx])
                score = item.get("relevance_score")
                if score is not None:
                    doc["rerank_score"] = score
                reranked.append(doc)

        if reranked:
            result = reranked[:rerank_top_n]
            meta["rerank_output_count"] = len(result)
            if rerank_cache_key:
                _store_rerank_result(rerank_cache_key, result, docs_for_rerank)
            return result, meta

        meta["rerank_error"] = "empty_rerank_results"
        result = docs_for_rerank[:rerank_top_n]
        meta["rerank_output_count"] = len(result)
        return result, meta
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        meta["rerank_error"] = str(e)
        result = docs_for_rerank[:rerank_top_n]
        meta["rerank_output_count"] = len(result)
        return result, meta


def _get_stepback_model():
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请将用户的具体问题抽象成更高层次、更概括的‘退步问题’，"
        "用于探寻背后的通用原理或核心概念。只输出退步问题一句话，不要解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请简要回答以下退步问题，提供通用原理/背景知识，"
        "控制在120字以内。只输出答案，不要列出推理过程。\n"
        f"退步问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _extract_json_object(text: str) -> dict:
    content = (text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


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
        payload = _extract_json_object(getattr(response, "content", response) or "")
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
    clean_files = []
    seen = set()
    for filename in filenames or []:
        name = (filename or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        clean_files.append(_escape_milvus_string(name))
    if not clean_files:
        return ""
    quoted = ", ".join(f'"{filename}"' for filename in clean_files)
    return f"filename in [{quoted}]"


def _dedupe_docs(docs: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for item in docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


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
    filename_registry = get_filename_registry(_milvus_manager, cache)
    query_plan = parse_query_plan(
        raw_query=query,
        filename_registry=filename_registry,
        context_files=context_files,
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

    if query_plan.scope_mode == "filter" and routable_matched_files and not filename_filter:
        # Build scoped filter
        matched_filenames = [f for f, _ in routable_matched_files]
        scoped_filenames_quoted = ", ".join([f'"{f}"' for f in matched_filenames])
        scoped_filter = f"{base_filter} and filename in [{scoped_filenames_quoted}]"

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

            # Weighted RRF merge: scoped 80%, global 20%
            retrieved = _weighted_rrf_merge(
                [(scoped, 1.0 - DOC_SCOPE_GLOBAL_RESERVE_WEIGHT), (global_, DOC_SCOPE_GLOBAL_RESERVE_WEIGHT)],
                rrf_k=MILVUS_RRF_K,
            )

            scope_trace = {
                "query_plan": query_plan.to_dict(),
                "semantic_query": query_plan.semantic_query,
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
                "semantic_query": query_plan.semantic_query,
                "scope_mode": query_plan.scope_mode,
                "query_route": query_plan.route,
                "scope_filter_applied": False,
                "filename_boost_applied": False,
                "embedding_failed": True,
            }

        # Apply heading lexical scoring if enabled
        if HEADING_LEXICAL_ENABLED and query_plan.scope_mode in {"filter", "boost"} and query_plan.heading_hint:
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
        "semantic_query": query_plan.semantic_query,
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

        if query_plan.scope_mode == "boost":
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
            dense_embeddings = _embedding_service.get_embeddings([query])
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

            if query_plan.scope_mode == "boost":
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
