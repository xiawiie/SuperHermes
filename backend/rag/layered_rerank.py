"""Layered candidate strategy: split retrieval orchestration and L1 prefilter helpers.

The final CrossEncoder and structure rerank live in the shared retrieval pipeline.
This module only shapes the candidate pool before that shared post-processing.
The filename is kept for compatibility with existing imports and flags.
"""
from __future__ import annotations

from collections import defaultdict

from backend.rag.runtime_config import LayeredRerankConfig


# --- Backward-compatible default constants ---
_DEFAULT_CONFIG = LayeredRerankConfig()

LAYERED_RERANK_ENABLED = _DEFAULT_CONFIG.enabled
L0_DENSE_TOP_K = _DEFAULT_CONFIG.l0_dense_top_k
L0_SPARSE_TOP_K = _DEFAULT_CONFIG.l0_sparse_top_k
L0_HYBRID_GUARANTEE_K = _DEFAULT_CONFIG.l0_hybrid_guarantee_k
L0_FALLBACK_POOL_MIN = _DEFAULT_CONFIG.l0_fallback_pool_min

L1_TOP_FILES = _DEFAULT_CONFIG.l1_top_files
L1_CHUNKS_PER_FILE_DEFAULT = _DEFAULT_CONFIG.l1_chunks_per_file_default
L1_CHUNKS_PER_FILE_TOP3 = _DEFAULT_CONFIG.l1_chunks_per_file_top3
L1_CHUNKS_PER_SCOPE_FILE = _DEFAULT_CONFIG.l1_chunks_per_scope_file
L1_CHUNK_MARGIN_THRESHOLD = _DEFAULT_CONFIG.l1_chunk_margin_threshold
L1_ROUTE_GUARANTEE_K = _DEFAULT_CONFIG.l1_route_guarantee_k
L1_SLOT_C_MAX = _DEFAULT_CONFIG.l1_slot_c_max
L1_SLOT_A_MIN = _DEFAULT_CONFIG.l1_slot_a_min
L1_SLOT_B_MIN = _DEFAULT_CONFIG.l1_slot_b_min
L1_MIN_CANDIDATES = _DEFAULT_CONFIG.l1_min_candidates
L1_MAX_CANDIDATES = _DEFAULT_CONFIG.l1_max_candidates

L2_CE_HIGH_CONF_K = _DEFAULT_CONFIG.l2_ce_high_conf_k
L2_CE_DEFAULT_K = _DEFAULT_CONFIG.l2_ce_default_k
L2_CE_LOW_CONF_K = _DEFAULT_CONFIG.l2_ce_low_conf_k
L2_CE_TOP_N = _DEFAULT_CONFIG.l2_ce_top_n
L2_CE_TOP_N_LOW_CONF = _DEFAULT_CONFIG.l2_ce_top_n_low_conf

L3_ROOT_WEIGHT = _DEFAULT_CONFIG.l3_root_weight
L3_SAME_ROOT_CAP_DEFAULT = _DEFAULT_CONFIG.l3_same_root_cap_default
L3_SAME_ROOT_CAP_SCOPE_QUERY = _DEFAULT_CONFIG.l3_same_root_cap_scope_query
L3_SAME_ROOT_CAP_BROAD_QUERY = _DEFAULT_CONFIG.l3_same_root_cap_broad_query
L3_PROTECT_CE_TOP3 = _DEFAULT_CONFIG.l3_protect_ce_top3


# --- L1 Score Functions ---

def rank_score(rank: int | None, top_k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 - (rank - 1) / max(top_k - 1, 1)


def l1_chunk_score(
    dense_rank: int | None,
    sparse_rank: int | None,
    top_k: int,
    scope_score: float = 0.0,
    metadata_score: float = 0.0,
    anchor_score: float = 0.0,
) -> float:
    return (
        0.35 * rank_score(dense_rank, top_k)
        + 0.35 * rank_score(sparse_rank, top_k)
        + 0.15 * scope_score
        + 0.10 * metadata_score
        + 0.05 * anchor_score
    )


def _metadata_score(doc: dict) -> float:
    score = 0.0
    if doc.get("section_title"):
        score += 0.4
    if doc.get("section_path"):
        score += 0.3
    if doc.get("page_number"):
        score += 0.3
    return min(1.0, score)


def _anchor_score(doc: dict, anchor_chunk_ids: set[str]) -> float:
    return 1.0 if doc.get("chunk_id", "") in anchor_chunk_ids else 0.0


def file_aggregate_score(chunk_scores: list[float]) -> float:
    if not chunk_scores:
        return 0.0
    sorted_scores = sorted(chunk_scores, reverse=True)
    top3 = sorted_scores[:3]
    return max(sorted_scores) + 0.30 * (sum(top3) / len(top3))


def _per_file_cap(
    file_rank: int,
    is_scope: bool,
    in_file_margin: float,
    config: LayeredRerankConfig,
) -> int:
    if is_scope:
        return config.l1_chunks_per_scope_file
    if file_rank <= 3:
        return config.l1_chunks_per_file_top3
    if in_file_margin < config.l1_chunk_margin_threshold:
        return config.l1_chunks_per_file_top3
    return config.l1_chunks_per_file_default


def build_l1_candidates(
    candidates: list[dict],
    scope_matched_files: list[str],
    anchor_chunk_ids: list[str],
    min_k: int | None = None,
    max_k: int | None = None,
    config: LayeredRerankConfig | None = None,
) -> list[dict]:
    """Build L1 candidate set using 3-slot architecture."""
    config = config or _DEFAULT_CONFIG
    min_k = min_k or config.l1_min_candidates
    max_k = max_k or config.l1_max_candidates
    anchor_set = set(anchor_chunk_ids)
    scope_set = set(scope_matched_files)
    top_k = config.l0_dense_top_k

    # Compute L1 scores for all candidates
    scored: list[dict] = []
    for doc in candidates:
        s = l1_chunk_score(
            dense_rank=doc.get("dense_rank"),
            sparse_rank=doc.get("sparse_rank"),
            top_k=top_k,
            scope_score=doc.get("doc_scope_match_score", 0.0),
            metadata_score=_metadata_score(doc),
            anchor_score=_anchor_score(doc, anchor_set),
        )
        doc = dict(doc)
        doc["l1_score"] = s
        scored.append(doc)

    # --- Slot C: Guarantee (anchor + scope) ---
    slot_c: list[dict] = []
    seen_c: set[str] = set()

    for doc in scored:
        cid = doc.get("chunk_id", "")
        if cid in anchor_set and cid not in seen_c:
            slot_c.append(doc)
            seen_c.add(cid)

    scope_by_file: dict[str, list[dict]] = defaultdict(list)
    for doc in scored:
        fn = doc.get("filename", "")
        if fn in scope_set:
            scope_by_file[fn].append(doc)

    for fn, docs in scope_by_file.items():
        docs.sort(key=lambda d: d["l1_score"], reverse=True)
        for doc in docs[: config.l1_chunks_per_scope_file]:
            cid = doc.get("chunk_id", "")
            if cid not in seen_c:
                slot_c.append(doc)
                seen_c.add(cid)

    slot_c = slot_c[: config.l1_slot_c_max]

    # --- Slot A: File-aware ---
    by_file: dict[str, list[dict]] = defaultdict(list)
    for doc in scored:
        by_file[doc.get("filename", "")].append(doc)

    file_scores: list[tuple[str, float]] = []
    for fn, docs in by_file.items():
        fs = file_aggregate_score([d["l1_score"] for d in docs])
        file_scores.append((fn, fs))
    file_scores.sort(key=lambda x: x[1], reverse=True)

    slot_a: list[dict] = []
    seen_a: set[str] = set(seen_c)

    for file_rank, (fn, _fs) in enumerate(file_scores[: config.l1_top_files], 1):
        docs = by_file[fn]
        docs.sort(key=lambda d: d["l1_score"], reverse=True)
        is_scope = fn in scope_set

        margin = 1.0
        if len(docs) >= 2:
            margin = abs(docs[0]["l1_score"] - docs[min(3, len(docs) - 1)]["l1_score"])

        cap = _per_file_cap(file_rank, is_scope, margin, config)
        for doc in docs[:cap]:
            cid = doc.get("chunk_id", "")
            if cid not in seen_a:
                slot_a.append(doc)
                seen_a.add(cid)

    # --- Slot B: Route guarantee (dense-only, sparse-only, metadata) ---
    slot_b: list[dict] = []
    seen_b = set(seen_a)

    dense_only = [d for d in scored if d.get("in_dense") and not d.get("in_sparse")]
    sparse_only = [d for d in scored if d.get("in_sparse") and not d.get("in_dense")]
    meta_matched = [d for d in scored if _metadata_score(d) > 0.5]

    for bucket in [dense_only, sparse_only, meta_matched]:
        bucket.sort(key=lambda d: d["l1_score"], reverse=True)
        count = 0
        for doc in bucket:
            cid = doc.get("chunk_id", "")
            if cid not in seen_b and count < config.l1_route_guarantee_k:
                slot_b.append(doc)
                seen_b.add(cid)
                count += 1

    # --- Merge and enforce quotas ---
    all_selected = slot_c + slot_a + slot_b

    seen_final: set[str] = set()
    deduped: list[dict] = []
    for doc in all_selected:
        cid = doc.get("chunk_id", "")
        if cid not in seen_final:
            deduped.append(doc)
            seen_final.add(cid)

    deduped.sort(key=lambda d: d["l1_score"], reverse=True)

    if len(deduped) > max_k:
        deduped = deduped[:max_k]

    return deduped


# --- L2 Adaptive K ---

def select_ce_k(
    candidates: list[dict],
    scope_mode: str,
    exact_file_match: bool,
    is_ambiguous: bool,
    dense_sparse_disagree: bool,
    config: LayeredRerankConfig | None = None,
) -> tuple[int, int]:
    """Return (ce_input_k, ce_top_n)."""
    config = config or _DEFAULT_CONFIG
    if is_ambiguous or dense_sparse_disagree:
        return config.l2_ce_low_conf_k, config.l2_ce_top_n_low_conf

    top_margin = 0.0
    if len(candidates) >= 2:
        top_margin = abs(candidates[0].get("l1_score", 0) - candidates[1].get("l1_score", 0))

    if scope_mode == "filter" and exact_file_match and top_margin > 0.15:
        return config.l2_ce_high_conf_k, config.l2_ce_top_n

    if len(candidates) >= 2:
        dense_agree = (
            candidates[0].get("dense_rank") is not None
            and candidates[1].get("dense_rank") is not None
        )
        sparse_agree = (
            candidates[0].get("sparse_rank") is not None
            and candidates[1].get("sparse_rank") is not None
        )
        if (dense_agree or sparse_agree) and top_margin > 0.20:
            return config.l2_ce_default_k, config.l2_ce_top_n

    return config.l2_ce_default_k, config.l2_ce_top_n


# --- L3 Helpers ---

def select_root_cap(
    scope_mode: str,
    exact_file_match: bool,
    dominant_root_share: float,
    query_is_broad: bool,
    config: LayeredRerankConfig | None = None,
) -> int:
    config = config or _DEFAULT_CONFIG
    if scope_mode in ("filter", "boost") and exact_file_match:
        return config.l3_same_root_cap_scope_query
    if dominant_root_share > 0.8 and query_is_broad:
        return config.l3_same_root_cap_broad_query
    return config.l3_same_root_cap_default
