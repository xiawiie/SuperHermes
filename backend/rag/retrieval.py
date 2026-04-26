from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any


def weighted_rrf_merge(
    result_sets: list[tuple[list[dict], float]],
    rrf_k: int = 60,
) -> list[dict]:
    """Merge multiple retrieval result sets using weighted reciprocal-rank fusion."""
    scores: dict[str, float] = defaultdict(float)
    doc_by_id: dict[str, dict] = {}

    for docs, weight in result_sets:
        for rank_idx, doc in enumerate(docs, 1):
            chunk_id = str(doc.get("chunk_id") or doc.get("id") or "")
            if not chunk_id:
                continue
            scores[chunk_id] += weight / (rrf_k + rank_idx)
            doc_by_id.setdefault(chunk_id, doc)

    result = []
    for chunk_id in sorted(scores, key=lambda item: -scores[item]):
        doc = dict(doc_by_id[chunk_id])
        doc["rrf_merged_score"] = round(scores[chunk_id], 6)
        result.append(doc)
    return result


def doc_filename(doc: dict) -> str:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    return str(doc.get("filename") or metadata.get("filename") or "")


def apply_filename_boost(
    query_plan: Any,
    candidates: list[dict],
    *,
    doc_scope_match_boost: float,
    filename_normalizer,
    milvus_rrf_k: int,
    filename_boost_weight: float,
) -> list[dict]:
    """Soft-rank candidates from matched files in boost mode without hard filtering."""
    if query_plan.scope_mode != "boost" or not candidates:
        return candidates

    matched_scores = {
        str(filename): score
        for filename, score in query_plan.matched_files
        if score >= doc_scope_match_boost
    }
    if not matched_scores:
        return candidates

    normalized_scores = {
        filename_normalizer(filename): score
        for filename, score in matched_scores.items()
    }

    scored: list[tuple[float, dict]] = []
    for idx, doc in enumerate(candidates, 1):
        filename = doc_filename(doc)
        match_score = matched_scores.get(filename, 0.0)
        if not match_score:
            match_score = normalized_scores.get(filename_normalizer(filename), 0.0)

        rank_score = 1.0 / (milvus_rrf_k + idx)
        boosted_score = rank_score + (filename_boost_weight * match_score)
        next_doc = dict(doc)
        if match_score:
            next_doc["filename_boost_applied"] = True
            next_doc["filename_boost_match_score"] = round(match_score, 6)
            next_doc["filename_boost_score"] = round(boosted_score, 6)
        scored.append((boosted_score, next_doc))

    scored.sort(key=lambda item: -item[0])
    return [doc for _, doc in scored]


def apply_heading_lexical_scoring(
    query_plan: Any,
    candidates: list[dict],
    *,
    heading_lexical_weight: float,
    milvus_rrf_k: int,
) -> list[dict]:
    """Apply heading/section lexical scoring to candidates."""
    if not query_plan.heading_hint or query_plan.scope_mode not in {"filter", "boost"}:
        return candidates

    semantic_query = query_plan.semantic_query
    anchors = query_plan.anchors
    scored_candidates = []
    for idx, doc in enumerate(candidates, 1):
        section_path = str(doc.get("section_path") or "")
        heading = str(doc.get("section_title") or "")
        anchor_id = str(doc.get("anchor_id") or "")

        heading_lexical_score = (
            0.5 * SequenceMatcher(None, semantic_query, section_path).ratio()
            + 0.3 * SequenceMatcher(None, semantic_query, heading).ratio()
            + 0.2 * (1.0 if any(a in anchor_id or a in heading or a in section_path for a in anchors) else 0.0)
        )
        rrf_rank_normalized = 1.0 / (milvus_rrf_k + idx)
        final_sort_key = (1 - heading_lexical_weight) * rrf_rank_normalized + (
            heading_lexical_weight * heading_lexical_score
        )
        scored_candidates.append((final_sort_key, doc))

    scored_candidates.sort(key=lambda item: -item[0])
    return [doc for _, doc in scored_candidates]


def annotate_scope_scores(docs: list[dict], matched_files: list[tuple[str, float]]) -> list[dict]:
    if not docs or not matched_files:
        return docs
    by_filename = {filename: score for filename, score in matched_files}
    out = []
    for doc in docs:
        next_doc = dict(doc)
        score = by_filename.get(doc_filename(next_doc))
        if score is not None:
            next_doc["doc_scope_match_score"] = round(float(score), 6)
        out.append(next_doc)
    return out


def build_filename_filter(filenames: list[str] | None, *, escape_string) -> str:
    clean_files = []
    seen = set()
    for filename in filenames or []:
        name = (filename or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        clean_files.append(escape_string(name))
    if not clean_files:
        return ""
    quoted = ", ".join(f'"{filename}"' for filename in clean_files)
    return f"filename in [{quoted}]"


def dedupe_docs(docs: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for item in docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
