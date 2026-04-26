from __future__ import annotations

from collections import defaultdict
from typing import Any


def merge_to_parent_level(docs: list[dict], *, parent_store_getter, threshold: int = 2) -> tuple[list[dict], int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    parent_docs = parent_store_getter().get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: list[dict] = []
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

    deduped: list[dict] = []
    seen = set()
    for item in merged_docs:
        key = item.get("chunk_id") or (item.get("filename"), item.get("page_number"), item.get("text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped, merged_count


def auto_merge_documents(
    docs: list[dict],
    top_k: int,
    *,
    enabled: bool,
    threshold: int,
    parent_store_getter,
) -> tuple[list[dict], dict[str, Any]]:
    if not enabled or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": enabled,
            "auto_merge_applied": False,
            "auto_merge_threshold": threshold,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    merged_docs, merged_count = merge_to_parent_level(
        docs,
        parent_store_getter=parent_store_getter,
        threshold=threshold,
    )
    merged_docs.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    merged_docs = merged_docs[:top_k]

    return merged_docs, {
        "auto_merge_enabled": enabled,
        "auto_merge_applied": merged_count > 0,
        "auto_merge_threshold": threshold,
        "auto_merge_replaced_chunks": merged_count,
        "auto_merge_steps": int(merged_count > 0),
        "auto_merge_path": "L3->L1",
    }


def apply_structure_rerank(
    docs: list[dict],
    top_k: int,
    *,
    enabled: bool,
    root_weight: float,
    same_root_cap: int,
) -> tuple[list[dict], dict[str, Any]]:
    if not enabled:
        limited = docs[:top_k]
        return limited, {
            "structure_rerank_enabled": False,
            "structure_rerank_applied": False,
            "structure_rerank_root_weight": root_weight,
            "same_root_cap": same_root_cap,
            "dominant_root_id": None,
            "dominant_root_share": 0.0,
            "dominant_root_support": 0,
        }
    if not docs:
        return [], {
            "structure_rerank_enabled": enabled,
            "structure_rerank_applied": False,
            "structure_rerank_root_weight": root_weight,
            "same_root_cap": same_root_cap,
            "dominant_root_id": None,
            "dominant_root_share": 0.0,
            "dominant_root_support": 0,
        }

    grouped: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        if not root_id:
            root_id = f"__orphan__:{len(grouped)}"
        grouped[root_id].append(doc)

    root_scores: dict[str, float] = {}
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
        final_score = (1.0 - root_weight) * leaf_score + root_weight * root_score
        enriched = dict(doc)
        enriched["leaf_score"] = leaf_score
        enriched["root_score"] = root_score
        enriched["final_score"] = final_score
        scored_docs.append(enriched)

    scored_docs.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
    limited: list[dict] = []
    per_root: dict[str, int] = defaultdict(int)
    for doc in scored_docs:
        root_id = (doc.get("root_chunk_id") or doc.get("chunk_id") or "").strip()
        if per_root[root_id] >= same_root_cap:
            continue
        limited.append(doc)
        per_root[root_id] += 1
        if len(limited) >= top_k:
            break

    root_total_scores: dict[str, float] = defaultdict(float)
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
        "structure_rerank_enabled": enabled,
        "structure_rerank_applied": True,
        "structure_rerank_root_weight": root_weight,
        "same_root_cap": same_root_cap,
        "dominant_root_id": dominant_root_id,
        "dominant_root_share": dominant_root_share,
        "dominant_root_support": dominant_root_support,
    }
