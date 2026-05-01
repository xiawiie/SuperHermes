"""Layered candidate strategy helpers.

This module only owns L0/L1 candidate shaping. Shared L2 rerank and shared L3
postprocess stay in the retrieval pipeline.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class LayeredCandidatePreset:
    l0_dense_top_k: int = 80
    l0_sparse_top_k: int = 80
    l0_hybrid_guarantee_k: int = 20
    l0_fallback_pool_min: int = 60
    l1_top_files: int = 12
    l1_chunks_per_file_default: int = 3
    l1_chunks_per_file_top3: int = 4
    l1_chunks_per_scope_file: int = 6
    l1_chunk_margin_threshold: float = 0.05
    l1_route_guarantee_k: int = 5
    l1_slot_c_max: int = 20
    l1_slot_a_min: int = 18
    l1_slot_b_min: int = 6
    l1_min_candidates: int = 30
    l1_max_candidates: int = 40


DEFAULT_LAYERED_CANDIDATE_PRESET = LayeredCandidatePreset()


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
    config: LayeredCandidatePreset,
) -> int:
    if is_scope:
        return config.l1_chunks_per_scope_file
    if file_rank <= 3:
        return config.l1_chunks_per_file_top3
    if in_file_margin < config.l1_chunk_margin_threshold:
        return config.l1_chunks_per_file_top3
    return config.l1_chunks_per_file_default


def build_layered_l1_candidates(
    candidates: list[dict],
    scope_matched_files: list[str],
    anchor_chunk_ids: list[str],
    min_k: int | None = None,
    max_k: int | None = None,
    config: LayeredCandidatePreset | None = None,
) -> list[dict]:
    """Build the L1 candidate pool using file-aware slots."""
    config = config or DEFAULT_LAYERED_CANDIDATE_PRESET
    min_k = min_k or config.l1_min_candidates
    max_k = max_k or config.l1_max_candidates
    anchor_set = set(anchor_chunk_ids)
    scope_set = set(scope_matched_files)
    top_k = config.l0_dense_top_k

    scored: list[dict] = []
    for doc in candidates:
        score = l1_chunk_score(
            dense_rank=doc.get("dense_rank"),
            sparse_rank=doc.get("sparse_rank"),
            top_k=top_k,
            scope_score=doc.get("doc_scope_match_score", 0.0),
            metadata_score=_metadata_score(doc),
            anchor_score=_anchor_score(doc, anchor_set),
        )
        doc = dict(doc)
        doc["l1_score"] = score
        scored.append(doc)

    slot_c: list[dict] = []
    seen_c: set[str] = set()

    for doc in scored:
        candidate_id = doc.get("chunk_id", "")
        if candidate_id in anchor_set and candidate_id not in seen_c:
            slot_c.append(doc)
            seen_c.add(candidate_id)

    scope_by_file: dict[str, list[dict]] = defaultdict(list)
    for doc in scored:
        filename = doc.get("filename", "")
        if filename in scope_set:
            scope_by_file[filename].append(doc)

    for docs in scope_by_file.values():
        docs.sort(key=lambda item: item["l1_score"], reverse=True)
        for doc in docs[: config.l1_chunks_per_scope_file]:
            candidate_id = doc.get("chunk_id", "")
            if candidate_id not in seen_c:
                slot_c.append(doc)
                seen_c.add(candidate_id)

    slot_c = slot_c[: config.l1_slot_c_max]

    by_file: dict[str, list[dict]] = defaultdict(list)
    for doc in scored:
        by_file[doc.get("filename", "")].append(doc)

    file_scores = [
        (filename, file_aggregate_score([doc["l1_score"] for doc in docs]))
        for filename, docs in by_file.items()
    ]
    file_scores.sort(key=lambda item: item[1], reverse=True)

    slot_a: list[dict] = []
    seen_a: set[str] = set(seen_c)

    for file_rank, (filename, _score) in enumerate(file_scores[: config.l1_top_files], 1):
        docs = by_file[filename]
        docs.sort(key=lambda item: item["l1_score"], reverse=True)
        is_scope = filename in scope_set

        margin = 1.0
        if len(docs) >= 2:
            margin = abs(docs[0]["l1_score"] - docs[min(3, len(docs) - 1)]["l1_score"])

        cap = _per_file_cap(file_rank, is_scope, margin, config)
        for doc in docs[:cap]:
            candidate_id = doc.get("chunk_id", "")
            if candidate_id not in seen_a:
                slot_a.append(doc)
                seen_a.add(candidate_id)

    slot_b: list[dict] = []
    seen_b = set(seen_a)

    dense_only = [doc for doc in scored if doc.get("in_dense") and not doc.get("in_sparse")]
    sparse_only = [doc for doc in scored if doc.get("in_sparse") and not doc.get("in_dense")]
    meta_matched = [doc for doc in scored if _metadata_score(doc) > 0.5]

    for bucket in (dense_only, sparse_only, meta_matched):
        bucket.sort(key=lambda item: item["l1_score"], reverse=True)
        count = 0
        for doc in bucket:
            candidate_id = doc.get("chunk_id", "")
            if candidate_id not in seen_b and count < config.l1_route_guarantee_k:
                slot_b.append(doc)
                seen_b.add(candidate_id)
                count += 1

    selected = slot_c + slot_a + slot_b
    seen_final: set[str] = set()
    deduped: list[dict] = []
    for doc in selected:
        candidate_id = doc.get("chunk_id", "")
        if candidate_id not in seen_final:
            deduped.append(doc)
            seen_final.add(candidate_id)

    deduped.sort(key=lambda item: item["l1_score"], reverse=True)

    if len(deduped) < min_k:
        selected_ids = {doc.get("chunk_id", "") for doc in deduped}
        for doc in sorted(scored, key=lambda item: item["l1_score"], reverse=True):
            candidate_id = doc.get("chunk_id", "")
            if candidate_id in selected_ids:
                continue
            deduped.append(doc)
            selected_ids.add(candidate_id)
            if len(deduped) >= min_k:
                break

    return deduped[:max_k]
