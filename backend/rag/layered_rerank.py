"""Layered rerank: L0 split retrieval orchestration, L1 prefilter, L2 adaptive K, L3 helpers."""
from __future__ import annotations

import os
from collections import defaultdict


# --- Configuration ---
LAYERED_RERANK_ENABLED = os.getenv("LAYERED_RERANK_ENABLED", "false").lower() == "true"

L0_DENSE_TOP_K = int(os.getenv("L0_DENSE_TOP_K", "80"))
L0_SPARSE_TOP_K = int(os.getenv("L0_SPARSE_TOP_K", "80"))
L0_HYBRID_GUARANTEE_K = int(os.getenv("L0_HYBRID_GUARANTEE_K", "20"))
L0_FALLBACK_POOL_MIN = int(os.getenv("L0_FALLBACK_HYBRID_WHEN_POOL_LT", "60"))

L1_TOP_FILES = int(os.getenv("L1_TOP_FILES", "12"))
L1_CHUNKS_PER_FILE_DEFAULT = int(os.getenv("L1_CHUNKS_PER_FILE_DEFAULT", "3"))
L1_CHUNKS_PER_FILE_TOP3 = int(os.getenv("L1_CHUNKS_PER_FILE_TOP3", "4"))
L1_CHUNKS_PER_SCOPE_FILE = int(os.getenv("L1_CHUNKS_PER_SCOPE_FILE", "6"))
L1_CHUNK_MARGIN_THRESHOLD = float(os.getenv("L1_CHUNK_MARGIN_THRESHOLD", "0.05"))
L1_ROUTE_GUARANTEE_K = int(os.getenv("L1_ROUTE_GUARANTEE_K", "5"))
L1_SLOT_C_MAX = int(os.getenv("L1_SLOT_C_MAX", "20"))
L1_SLOT_A_MIN = int(os.getenv("L1_SLOT_A_MIN", "18"))
L1_SLOT_B_MIN = int(os.getenv("L1_SLOT_B_MIN", "6"))
L1_MIN_CANDIDATES = int(os.getenv("L1_MIN_CANDIDATES", "30"))
L1_MAX_CANDIDATES = int(os.getenv("L1_MAX_CANDIDATES", "40"))

L2_CE_HIGH_CONF_K = int(os.getenv("L2_CE_HIGH_CONF_K", "25"))
L2_CE_DEFAULT_K = int(os.getenv("L2_CE_DEFAULT_K", "32"))
L2_CE_LOW_CONF_K = int(os.getenv("L2_CE_LOW_CONF_K", "40"))
L2_CE_TOP_N = int(os.getenv("L2_CE_TOP_N", "15"))
L2_CE_TOP_N_LOW_CONF = int(os.getenv("L2_CE_TOP_N_LOW_CONF", "20"))

L3_ROOT_WEIGHT = float(os.getenv("L3_ROOT_WEIGHT", "0.15"))
L3_SAME_ROOT_CAP_DEFAULT = int(os.getenv("L3_SAME_ROOT_CAP_DEFAULT", "3"))
L3_SAME_ROOT_CAP_SCOPE_QUERY = int(os.getenv("L3_SAME_ROOT_CAP_SCOPE_QUERY", "5"))
L3_SAME_ROOT_CAP_BROAD_QUERY = int(os.getenv("L3_SAME_ROOT_CAP_BROAD_QUERY", "2"))
L3_PROTECT_CE_TOP3 = os.getenv("L3_PROTECT_CE_TOP3", "true").lower() == "true"


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


def _per_file_cap(file_rank: int, is_scope: bool, in_file_margin: float) -> int:
    if is_scope:
        return L1_CHUNKS_PER_SCOPE_FILE
    if file_rank <= 3:
        return L1_CHUNKS_PER_FILE_TOP3
    if in_file_margin < L1_CHUNK_MARGIN_THRESHOLD:
        return L1_CHUNKS_PER_FILE_TOP3
    return L1_CHUNKS_PER_FILE_DEFAULT


def build_l1_candidates(
    candidates: list[dict],
    scope_matched_files: list[str],
    anchor_chunk_ids: list[str],
    min_k: int | None = None,
    max_k: int | None = None,
) -> list[dict]:
    """Build L1 candidate set using 3-slot architecture."""
    min_k = min_k or L1_MIN_CANDIDATES
    max_k = max_k or L1_MAX_CANDIDATES
    anchor_set = set(anchor_chunk_ids)
    scope_set = set(scope_matched_files)
    top_k = L0_DENSE_TOP_K

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
        for doc in docs[:L1_CHUNKS_PER_SCOPE_FILE]:
            cid = doc.get("chunk_id", "")
            if cid not in seen_c:
                slot_c.append(doc)
                seen_c.add(cid)

    slot_c = slot_c[:L1_SLOT_C_MAX]

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

    for file_rank, (fn, _fs) in enumerate(file_scores[:L1_TOP_FILES], 1):
        docs = by_file[fn]
        docs.sort(key=lambda d: d["l1_score"], reverse=True)
        is_scope = fn in scope_set

        margin = 1.0
        if len(docs) >= 2:
            margin = abs(docs[0]["l1_score"] - docs[min(3, len(docs) - 1)]["l1_score"])

        cap = _per_file_cap(file_rank, is_scope, margin)
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
            if cid not in seen_b and count < L1_ROUTE_GUARANTEE_K:
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
) -> tuple[int, int]:
    """Return (ce_input_k, ce_top_n)."""
    if is_ambiguous or dense_sparse_disagree:
        return L2_CE_LOW_CONF_K, L2_CE_TOP_N_LOW_CONF

    top_margin = 0.0
    if len(candidates) >= 2:
        top_margin = abs(candidates[0].get("l1_score", 0) - candidates[1].get("l1_score", 0))

    if scope_mode == "filter" and exact_file_match and top_margin > 0.15:
        return L2_CE_HIGH_CONF_K, L2_CE_TOP_N

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
            return L2_CE_DEFAULT_K, L2_CE_TOP_N

    return L2_CE_DEFAULT_K, L2_CE_TOP_N


# --- L3 Helpers ---

def select_root_cap(
    scope_mode: str,
    exact_file_match: bool,
    dominant_root_share: float,
    query_is_broad: bool,
) -> int:
    if scope_mode in ("filter", "boost") and exact_file_match:
        return L3_SAME_ROOT_CAP_SCOPE_QUERY
    if dominant_root_share > 0.8 and query_is_broad:
        return L3_SAME_ROOT_CAP_BROAD_QUERY
    return L3_SAME_ROOT_CAP_DEFAULT
