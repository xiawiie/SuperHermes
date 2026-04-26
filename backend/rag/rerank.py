from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import requests


@dataclass(frozen=True)
class RerankRuntime:
    provider: str
    model: str | None
    binding_host: str | None
    api_key: str | None
    cpu_top_n_cap: int
    cache_enabled: bool
    pair_enrichment_enabled: bool
    score_fusion_enabled: bool
    fusion_weights: dict[str, float]
    milvus_rrf_k: int
    get_endpoint: Callable[[], str]
    effective_top_n: Callable[[int, int], int]
    effective_input_k: Callable[[int, int], tuple[int, str, int]]
    get_local_reranker: Callable[[], Any]
    cache_key: Callable[[str, list[dict], int, int, bool], str]
    load_cached_result: Callable[[str, list[dict], int], list[dict] | None]
    store_result: Callable[[str, list[dict], list[dict]], None]
    doc_text_getter: Callable[[dict], str]
    post: Callable[..., Any]


def build_enriched_pair(doc: dict, *, doc_text_getter) -> str:
    filename = str(doc.get("filename") or "")
    section_path = str(doc.get("section_path") or doc.get("section_title") or "")
    page = doc.get("page_number") or doc.get("page_start") or ""
    anchor = str(doc.get("anchor_id") or doc.get("anchor") or "")
    heading = str(doc.get("section_title") or "")
    body = doc_text_getter(doc)

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


def rerank_pair_text(doc: dict, enrichment_enabled: bool, *, doc_text_getter) -> str:
    if enrichment_enabled:
        return build_enriched_pair(doc, doc_text_getter=doc_text_getter)
    return doc_text_getter(doc)


def normalize_float_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def metadata_match_score(doc: dict) -> float:
    score = 0.0
    if doc.get("anchor_id"):
        score += 0.35
    if doc.get("section_title") or doc.get("section_path"):
        score += 0.35
    if doc.get("page_number") or doc.get("page_start"):
        score += 0.30
    return min(1.0, score)


def scope_match_score(doc: dict) -> float:
    for key in ("doc_scope_match_score", "filename_boost_match_score"):
        try:
            value = float(doc.get(key) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return min(1.0, max(0.0, value))
    return 0.0


def rerank_rrf_score(doc: dict, *, milvus_rrf_k: int) -> float:
    try:
        rank = int(doc.get("rrf_rank") or 0)
    except (TypeError, ValueError):
        rank = 0
    if rank <= 0:
        return 0.0
    return 1.0 / (milvus_rrf_k + rank)


def apply_rerank_score_fusion(
    indexed_scores: list[tuple[int, float]],
    docs_for_rerank: list[dict],
    *,
    enabled: bool,
    weights: dict[str, float],
    milvus_rrf_k: int,
) -> list[tuple[int, float]]:
    if not enabled or not indexed_scores:
        return sorted(indexed_scores, key=lambda item: item[1], reverse=True)

    ordered_indices = [idx for idx, _ in indexed_scores]
    rerank_norm = normalize_float_scores([float(score) for _, score in indexed_scores])
    rrf_norm = normalize_float_scores(
        [rerank_rrf_score(docs_for_rerank[idx], milvus_rrf_k=milvus_rrf_k) for idx in ordered_indices]
    )
    scope_scores = [scope_match_score(docs_for_rerank[idx]) for idx in ordered_indices]
    metadata_scores = [metadata_match_score(docs_for_rerank[idx]) for idx in ordered_indices]

    normalized_weights = {
        "rerank": max(0.0, weights.get("rerank", 0.0)),
        "rrf": max(0.0, weights.get("rrf", 0.0)),
        "scope": max(0.0, weights.get("scope", 0.0)),
        "metadata": max(0.0, weights.get("metadata", 0.0)),
    }
    total_weight = sum(normalized_weights.values()) or 1.0

    fused: list[tuple[int, float]] = []
    for pos, idx in enumerate(ordered_indices):
        final_score = (
            normalized_weights["rerank"] * rerank_norm[pos]
            + normalized_weights["rrf"] * rrf_norm[pos]
            + normalized_weights["scope"] * scope_scores[pos]
            + normalized_weights["metadata"] * metadata_scores[pos]
        ) / total_weight
        fused.append((idx, final_score))

    return sorted(fused, key=lambda item: item[1], reverse=True)


def _rank_from_scores(
    raw_indexed_scores: list[tuple[int, float]],
    raw_scores_by_idx: dict[int, float],
    docs_for_rerank: list[dict],
    rerank_top_n: int,
    runtime: RerankRuntime,
) -> list[dict]:
    indexed_scores = apply_rerank_score_fusion(
        raw_indexed_scores,
        docs_for_rerank,
        enabled=runtime.score_fusion_enabled,
        weights=runtime.fusion_weights,
        milvus_rrf_k=runtime.milvus_rrf_k,
    )
    reranked = []
    for idx, score in indexed_scores[:rerank_top_n]:
        doc = dict(docs_for_rerank[idx])
        doc["raw_rerank_score"] = raw_scores_by_idx.get(idx, float(score))
        if runtime.score_fusion_enabled:
            doc["fusion_score"] = float(score)
        doc["rerank_score"] = float(score)
        reranked.append(doc)
    return reranked


def rerank_documents(query: str, docs: list[dict], top_k: int, runtime: RerankRuntime) -> tuple[list[dict], dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    rerank_top_n = runtime.effective_top_n(top_k, len(docs_with_rank))
    rerank_input_k, rerank_input_device_tier, rerank_input_cap = runtime.effective_input_k(
        rerank_top_n,
        len(docs_with_rank),
    )
    rerank_top_n = min(rerank_top_n, rerank_input_k)
    docs_for_rerank = docs_with_rank[:rerank_input_k]
    is_local = runtime.provider == "local" and bool(runtime.model)
    is_ollama = runtime.provider == "ollama" and bool(runtime.model and runtime.binding_host)
    is_api = runtime.provider not in ("local", "ollama") and bool(
        runtime.model and runtime.api_key and runtime.binding_host
    )
    meta: dict[str, Any] = {
        "rerank_enabled": is_local or is_ollama or is_api,
        "rerank_applied": False,
        "rerank_model": runtime.model,
        "rerank_provider": runtime.provider,
        "rerank_endpoint": runtime.get_endpoint()
        if is_api
        else (f"{runtime.binding_host.rstrip('/')}/api/rerank" if is_ollama and runtime.binding_host else None),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
        "rerank_top_n": rerank_top_n,
        "rerank_cpu_top_n_cap": runtime.cpu_top_n_cap,
        "rerank_input_count": rerank_input_k if (is_local or is_ollama or is_api) else 0,
        "rerank_output_count": 0,
        "rerank_input_cap": rerank_input_cap,
        "rerank_input_device_tier": rerank_input_device_tier,
        "rerank_cache_enabled": runtime.cache_enabled,
        "rerank_cache_hit": False,
        "rerank_pair_enrichment_enabled": runtime.pair_enrichment_enabled,
        "rerank_score_fusion_enabled": runtime.score_fusion_enabled,
        "rerank_fusion_weights": runtime.fusion_weights,
    }
    if not docs_with_rank or not meta["rerank_enabled"]:
        result = docs_with_rank[:rerank_top_n]
        meta["rerank_output_count"] = len(result)
        return result, meta

    rerank_cache_key = ""
    if runtime.cache_enabled and docs_for_rerank and rerank_top_n > 0:
        rerank_cache_key = runtime.cache_key(
            query,
            docs_for_rerank,
            rerank_top_n,
            rerank_input_k,
            runtime.pair_enrichment_enabled,
        )
        cached_result = runtime.load_cached_result(rerank_cache_key, docs_for_rerank, rerank_top_n)
        if cached_result:
            meta["rerank_applied"] = True
            meta["rerank_cache_hit"] = True
            meta["rerank_output_count"] = len(cached_result)
            return cached_result, meta

    if is_local:
        try:
            reranker = runtime.get_local_reranker()
            if not reranker:
                meta["rerank_error"] = "local_reranker_not_loaded"
                result = docs_for_rerank[:rerank_top_n]
                meta["rerank_output_count"] = len(result)
                return result, meta
            texts = [
                rerank_pair_text(doc, runtime.pair_enrichment_enabled, doc_text_getter=runtime.doc_text_getter)
                for doc in docs_for_rerank
            ]
            scores = reranker.predict([[query, text] for text in texts])
            raw_scores = [float(score) for score in scores]
            reranked = _rank_from_scores(
                list(enumerate(raw_scores)),
                {idx: score for idx, score in enumerate(raw_scores)},
                docs_for_rerank,
                rerank_top_n,
                runtime,
            )
            meta["rerank_applied"] = True
            result = reranked[:rerank_top_n]
            meta["rerank_output_count"] = len(result)
            if rerank_cache_key:
                runtime.store_result(rerank_cache_key, result, docs_for_rerank)
            return result, meta
        except Exception as exc:
            meta["rerank_error"] = str(exc)
            result = docs_for_rerank[:rerank_top_n]
            meta["rerank_output_count"] = len(result)
            return result, meta

    payload = {
        "model": runtime.model,
        "query": query,
        "documents": [
            rerank_pair_text(doc, runtime.pair_enrichment_enabled, doc_text_getter=runtime.doc_text_getter)
            for doc in docs_for_rerank
        ],
        "top_n": rerank_input_k if runtime.score_fusion_enabled else rerank_top_n,
    }
    headers = None
    timeout = 30
    if is_api:
        payload["return_documents"] = False
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {runtime.api_key}",
        }
        timeout = 15

    try:
        meta["rerank_applied"] = True
        response = runtime.post(
            meta["rerank_endpoint"],
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        if response.status_code >= 400:
            meta["rerank_error"] = f"HTTP {response.status_code}: {response.text}"
            result = docs_for_rerank[:rerank_top_n]
            meta["rerank_output_count"] = len(result)
            return result, meta

        items = response.json().get("results", [])
        raw_indexed_scores = []
        raw_scores_by_idx = {}
        for item in items:
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(docs_for_rerank):
                score = item.get("relevance_score")
                try:
                    raw_score = float(score if score is not None else 0.0)
                except (TypeError, ValueError):
                    raw_score = 0.0
                raw_scores_by_idx[idx] = raw_score
                raw_indexed_scores.append((idx, raw_score))

        if raw_indexed_scores:
            result = _rank_from_scores(raw_indexed_scores, raw_scores_by_idx, docs_for_rerank, rerank_top_n, runtime)
            meta["rerank_output_count"] = len(result)
            if rerank_cache_key:
                runtime.store_result(rerank_cache_key, result, docs_for_rerank)
            return result, meta

        meta["rerank_error"] = "empty_rerank_results"
        result = docs_for_rerank[:rerank_top_n]
        meta["rerank_output_count"] = len(result)
        return result, meta
    except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        meta["rerank_error"] = str(exc)
        result = docs_for_rerank[:rerank_top_n]
        meta["rerank_output_count"] = len(result)
        return result, meta
