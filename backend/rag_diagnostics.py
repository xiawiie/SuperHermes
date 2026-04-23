"""Minimal RAG failure classification helpers."""
from __future__ import annotations

import re
from typing import Any, TypedDict


class DiagnosticResult(TypedDict):
    category: str
    failed_stage: str
    evidence: dict[str, Any]
    suggestions: list[str]


def _as_set(values: list[str] | None) -> set[str]:
    return {str(item) for item in values or [] if str(item)}


_NUMERAL_NE = r"(?<![一二三四五六七八九十百千万零两\d])"
_NUMERAL_NLA = r"(?![一二三四五六七八九十百千万零两\d])"


def _anchor_in_text(anchor: str, text: str) -> bool:
    """Match anchor as a discrete structural unit in free-form text.

    Prevents ``"1.2"`` matching ``"11.2"``, ``"一、"`` matching ``"二十一、"``.
    """
    if not anchor or not text:
        return False
    escaped = re.escape(anchor)
    return bool(re.search(_NUMERAL_NE + escaped + _NUMERAL_NLA, text))


def _matches_expected(
    docs: list[dict] | None,
    expected_chunk_ids: list[str] | None = None,
    expected_root_ids: list[str] | None = None,
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
) -> dict[str, Any]:
    docs = docs or []
    chunk_ids = _as_set(expected_chunk_ids)
    root_ids = _as_set(expected_root_ids)
    anchors = _as_set(expected_anchors)
    keywords = _as_set(expected_keywords)

    matched_chunk_ids = []
    matched_root_ids = []
    matched_anchors = []
    matched_keywords = []

    for doc in docs:
        chunk_id = str(doc.get("chunk_id") or "")
        root_id = str(doc.get("root_chunk_id") or "")
        anchor_id = str(doc.get("anchor_id") or "")
        section_title = str(doc.get("section_title") or "")
        section_path = str(doc.get("section_path") or "")
        text = str(doc.get("retrieval_text") or doc.get("text") or doc.get("text_preview") or "")

        if chunk_id and chunk_id in chunk_ids:
            matched_chunk_ids.append(chunk_id)
        if root_id and root_id in root_ids:
            matched_root_ids.append(root_id)
        for anchor in anchors:
            if anchor and (
                anchor == anchor_id
                or anchor in section_title
                or anchor in section_path
                or _anchor_in_text(anchor, text)
            ):
                matched_anchors.append(anchor)
        for keyword in keywords:
            if keyword and keyword in text:
                matched_keywords.append(keyword)

    return {
        "matched": bool(matched_chunk_ids or matched_root_ids or matched_anchors or matched_keywords),
        "matched_chunk_ids": sorted(set(matched_chunk_ids)),
        "matched_root_ids": sorted(set(matched_root_ids)),
        "matched_anchors": sorted(set(matched_anchors)),
        "matched_keywords": sorted(set(matched_keywords)),
    }


def _result(category: str, failed_stage: str, evidence: dict[str, Any], suggestions: list[str]) -> DiagnosticResult:
    return {
        "category": category,
        "failed_stage": failed_stage,
        "evidence": evidence,
        "suggestions": suggestions,
    }


def classify_failure(
    query: str,
    rag_trace: dict | None,
    expected_chunk_ids: list[str] | None = None,
    expected_root_ids: list[str] | None = None,
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
) -> DiagnosticResult:
    """Classify a RAG failure using the smallest reliable evidence set."""
    if not rag_trace or "retrieved_chunks" not in rag_trace:
        return _result(
            "insufficient_trace",
            "unknown",
            {"reason": "missing_rag_trace", "query": query},
            ["确保 RAG 主链路稳定输出 retrieved_chunks 和基础置信度字段。"],
        )

    has_ground_truth = bool(expected_chunk_ids or expected_root_ids or expected_anchors or expected_keywords)
    final_match = _matches_expected(
        rag_trace.get("retrieved_chunks", []),
        expected_chunk_ids=expected_chunk_ids,
        expected_root_ids=expected_root_ids,
        expected_anchors=expected_anchors,
        expected_keywords=expected_keywords,
    )
    if has_ground_truth and final_match["matched"]:
        return _result(
            "ok",
            "none",
            {
                "final_match": final_match,
                "fallback_required": bool(rag_trace.get("fallback_required")),
                "confidence_reasons": rag_trace.get("confidence_reasons") or [],
            },
            [],
        )

    if rag_trace.get("fallback_required") or rag_trace.get("confidence_reasons"):
        return _result(
            "low_confidence",
            "confidence_gate",
            {
                "fallback_required": bool(rag_trace.get("fallback_required")),
                "confidence_reasons": rag_trace.get("confidence_reasons") or [],
                "top_score": rag_trace.get("top_score"),
                "top_margin": rag_trace.get("top_margin"),
                "dominant_root_share": rag_trace.get("dominant_root_share"),
                "anchor_match": rag_trace.get("anchor_match"),
            },
            ["检查 top_margin、dominant_root_share 和 anchor_match，判断门控阈值是否过松或过紧。"],
        )

    if not has_ground_truth:
        return _result(
            "insufficient_trace",
            "unknown",
            {"reason": "missing_ground_truth", "query": query},
            ["线上无标准答案时只能做候选归因；离线评估请提供 expected_root_ids、expected_anchors 或 expected_keywords。"],
        )

    before = rag_trace.get("candidates_before_rerank")
    after_rerank = rag_trace.get("candidates_after_rerank")
    after_structure = rag_trace.get("candidates_after_structure_rerank")
    if before is None or after_rerank is None or after_structure is None:
        return _result(
            "insufficient_trace",
            "unknown",
            {
                "reason": "missing_candidate_stage_trace",
                "has_before_rerank": before is not None,
                "has_after_rerank": after_rerank is not None,
                "has_after_structure": after_structure is not None,
            },
            ["在评估模式下记录 candidates_before_rerank、candidates_after_rerank 和 candidates_after_structure_rerank。"],
        )

    before_match = _matches_expected(before, expected_chunk_ids, expected_root_ids, expected_anchors, expected_keywords)
    if not before_match["matched"]:
        return _result(
            "recall_miss",
            "recall",
            {"candidate_match": before_match},
            ["调整 chunking 边界、retrieval_text、dense/sparse candidate 范围，并检查 embedding/sparse 信号。"],
        )

    after_rerank_match = _matches_expected(after_rerank, expected_chunk_ids, expected_root_ids, expected_anchors, expected_keywords)
    if not after_rerank_match["matched"]:
        return _result(
            "ranking_miss",
            "rerank",
            {"before_rerank_match": before_match, "after_rerank_match": after_rerank_match},
            ["扩大 rerank 候选范围，检查 reranker 输入是否正确使用 retrieval_text。"],
        )

    after_structure_match = _matches_expected(after_structure, expected_chunk_ids, expected_root_ids, expected_anchors, expected_keywords)
    if not after_structure_match["matched"]:
        return _result(
            "ranking_miss",
            "structure_rerank",
            {"after_rerank_match": after_rerank_match, "after_structure_match": after_structure_match},
            ["调整 STRUCTURE_RERANK_ROOT_WEIGHT、SAME_ROOT_CAP，并检查是否被错误 root 压制。"],
        )

    return _result(
        "mixed_failure",
        "unknown",
        {
            "final_match": final_match,
            "before_rerank_match": before_match,
            "after_rerank_match": after_rerank_match,
            "after_structure_match": after_structure_match,
        },
        ["候选阶段曾命中但最终未命中；先复查 final top-k 组装，再检查排序与去重逻辑。"],
    )
