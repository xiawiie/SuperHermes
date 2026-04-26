from __future__ import annotations

from collections import defaultdict
from typing import Any


def extract_query_anchors(query: str, *, anchor_pattern) -> list[str]:
    if not query:
        return []
    return list(dict.fromkeys(anchor_pattern.findall(query)))


def doc_matches_anchor(doc: dict, anchor: str, *, doc_text_getter) -> bool:
    if not anchor:
        return False
    for value in (
        doc.get("anchor_id"),
        doc.get("section_title"),
        doc.get("section_path"),
    ):
        if anchor and anchor in str(value or ""):
            return True
    prefix = doc_text_getter(doc).split("\n", 1)[0]
    return anchor in prefix


def evaluate_retrieval_confidence(
    query: str,
    docs: list[dict],
    *,
    confidence_gate_enabled: bool,
    low_conf_top_margin: float,
    low_conf_root_share: float,
    low_conf_top_score: float,
    enable_anchor_gate: bool,
    anchor_pattern,
    doc_text_getter,
) -> dict[str, Any]:
    if not docs:
        return {
            "confidence_gate_enabled": confidence_gate_enabled,
            "fallback_required": confidence_gate_enabled,
            "confidence_reasons": ["no_docs"] if confidence_gate_enabled else [],
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

    root_total_scores: dict[str, float] = defaultdict(float)
    root_supports: dict[str, int] = defaultdict(int)
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

    anchors = extract_query_anchors(query, anchor_pattern=anchor_pattern) if enable_anchor_gate else []
    anchor_match = True
    if anchors:
        anchor_match = False
        for anchor in anchors:
            if any(doc_matches_anchor(doc, anchor, doc_text_getter=doc_text_getter) for doc in docs[:2]):
                anchor_match = True
                break

    reasons: list[str] = []
    if anchors and not anchor_match:
        reasons.append("anchor_mismatch")
    if top_margin < low_conf_top_margin and dominant_root_share < low_conf_root_share:
        reasons.append("weak_margin_and_root")
    if top_score < low_conf_top_score and top_margin < low_conf_top_margin:
        reasons.append("low_score_and_margin")

    if not confidence_gate_enabled:
        reasons = []

    return {
        "confidence_gate_enabled": confidence_gate_enabled,
        "fallback_required": bool(reasons),
        "confidence_reasons": reasons,
        "top_margin": top_margin,
        "top_score": top_score,
        "dominant_root_share": dominant_root_share,
        "dominant_root_support": dominant_root_support,
        "anchor_match": anchor_match,
        "query_anchors": anchors,
    }
