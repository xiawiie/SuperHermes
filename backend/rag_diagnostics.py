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


def _page_candidates(doc: dict) -> set[str]:
    pages: set[str] = set()

    def add_page(value: Any) -> int | None:
        try:
            page = int(value)
        except (TypeError, ValueError):
            return None
        pages.add(str(page))
        pages.add(str(page + 1))
        return page

    page_number = add_page(doc.get("page_number"))
    page_start = add_page(doc.get("page_start"))
    page_end = add_page(doc.get("page_end"))

    if page_start is not None and page_end is not None and page_end >= page_start:
        for page in range(page_start, min(page_end, page_start + 20) + 1):
            pages.add(str(page))
            pages.add(str(page + 1))
    elif page_number is None and page_start is None and page_end is None:
        return pages

    return pages


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
    expected_files: list[str] | None = None,
    expected_pages: list[str] | None = None,
    hard_negative_files: list[str] | None = None,
) -> DiagnosticResult:
    """Classify a RAG failure into one of five categories:
    - file_recall_miss: correct file not in candidates
    - page_miss: correct file in top5 but wrong page
    - ranking_miss: correct candidate exists but rerank dropped it
    - hard_negative_confusion: all top5 from hard negatives
    - low_confidence: top1 score below threshold
    """
    if not rag_trace or "retrieved_chunks" not in rag_trace:
        return _result(
            "insufficient_trace",
            "unknown",
            {"reason": "missing_rag_trace", "query": query},
            ["确保 RAG 主链路稳定输出 retrieved_chunks 和基础置信度字段。"],
        )

    has_ground_truth = bool(
        expected_chunk_ids
        or expected_root_ids
        or expected_anchors
        or expected_keywords
        or expected_files
        or expected_pages
    )

    # Check top5 for hard negative confusion
    top5_chunks = rag_trace.get("retrieved_chunks", [])[:5]
    top5_files = {str(c.get("filename") or "") for c in top5_chunks}
    hard_neg_set = _as_set(hard_negative_files)
    expected_file_set = _as_set(expected_files)

    # hard_negative_confusion: all top5 from hard negatives
    if hard_neg_set and top5_files and top5_files.issubset(hard_neg_set):
        return _result(
            "hard_negative_confusion",
            "rerank",
            {
                "top5_files": sorted(top5_files),
                "hard_negative_files": sorted(hard_neg_set),
                "expected_files": sorted(expected_file_set),
            },
            ["检查 reranker 是否能有效区分 hard negative 和正确文档；考虑添加 filename boost 或产品族惩罚。"],
        )

    # Check if correct file is in top5
    file_in_top5 = bool(expected_file_set & top5_files) if expected_file_set else None

    # Check if correct file is in candidates (before rerank)
    before = rag_trace.get("candidates_before_rerank") or []
    before_files = {str(c.get("filename") or "") for c in before}
    file_in_candidates = bool(expected_file_set & before_files) if expected_file_set else None

    # file_recall_miss: correct file not in candidates at all
    if expected_file_set and not file_in_candidates:
        return _result(
            "file_recall_miss",
            "recall",
            {
                "expected_files": sorted(expected_file_set),
                "top5_files": sorted(top5_files),
                "candidate_file_count": len(before_files),
            },
            ["调整 embedding/sparse 检索范围，检查 filename registry 和 query plan 路由。"],
        )

    # ranking_miss: correct file in candidates but not in top5
    if expected_file_set and file_in_candidates and not file_in_top5:
        return _result(
            "ranking_miss",
            "rerank",
            {
                "expected_files": sorted(expected_file_set),
                "file_in_candidates": True,
                "file_in_top5": False,
            },
            ["扩大 rerank 候选范围，检查 reranker 是否正确排序含正确文件的候选。"],
        )

    # page_miss: correct file in top5 but wrong page
    if expected_file_set and file_in_top5 and expected_pages:
        expected_pages_str = {str(p) for p in expected_pages}
        top5_pages = set()
        for c in top5_chunks:
            if str(c.get("filename") or "") in expected_file_set:
                top5_pages |= _page_candidates(c)
        if not (expected_pages_str & top5_pages):
            return _result(
                "page_miss",
                "rerank",
                {
                    "expected_files": sorted(expected_file_set),
                    "expected_pages": sorted(expected_pages_str),
                    "top5_pages_for_expected_files": sorted(top5_pages),
                },
                ["检查文档分页是否正确，chunk 是否包含页码元数据，reranker 是否给正确页码更高分数。"],
            )

    if expected_file_set and file_in_top5:
        return _result(
            "ok",
            "none",
            {
                "expected_files": sorted(expected_file_set),
                "top5_files": sorted(top5_files),
                "expected_pages": sorted(str(p) for p in expected_pages or []),
            },
            [],
        )

    # Legacy match checks for non-file-based ground truth
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

    # low_confidence: top1 score is very low
    top1_score = None
    for c in top5_chunks:
        s = c.get("score") or c.get("rerank_score") or c.get("final_score")
        if s is not None:
            top1_score = float(s)
            break
    if top1_score is not None and top1_score < 0.20:
        return _result(
            "low_confidence",
            "confidence_gate",
            {
                "top1_score": top1_score,
                "fallback_required": bool(rag_trace.get("fallback_required")),
                "confidence_reasons": rag_trace.get("confidence_reasons") or [],
            },
            ["检索结果置信度过低，检查 query 是否有效、embedding 是否退化、索引是否过期。"],
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
            ["线上无标准答案时只能做候选归因；离线评估请提供 expected_files、expected_anchors 或 expected_keywords。"],
        )

    # Root/chunk/anchor/keyword ground truth: attribute recall vs rerank when candidate lists exist
    if rag_trace.get("candidates_before_rerank") is None:
        return _result(
            "insufficient_trace",
            "unknown",
            {
                "reason": "missing_prerank_candidates",
                "query": query,
                "final_match": final_match,
            },
            ["补充 candidates_before_rerank / candidates_after_rerank 以便区分召回失败与排序失败。"],
        )

    before_match = _matches_expected(before, expected_chunk_ids, expected_root_ids, expected_anchors, expected_keywords)
    after_rerank = rag_trace.get("candidates_after_rerank") or []
    after_rerank_match = _matches_expected(after_rerank, expected_chunk_ids, expected_root_ids, expected_anchors, expected_keywords)

    if not before_match["matched"]:
        return _result(
            "file_recall_miss",
            "recall",
            {
                "final_match": final_match,
                "before_rerank_match": before_match,
                "after_rerank_match": after_rerank_match,
            },
            ["扩大检索召回（embedding/sparse/查询改写），确认黄金片段是否进入 rerank 前候选。"],
        )

    if before_match["matched"] and not final_match["matched"]:
        return _result(
            "ranking_miss",
            "rerank",
            {
                "final_match": final_match,
                "before_rerank_match": before_match,
                "after_rerank_match": after_rerank_match,
            },
            ["候选阶段曾命中但最终未命中；检查 rerank / structure_rerank 与 top-k 截断。"],
        )

    return _result(
        "ranking_miss",
        "unknown",
        {
            "final_match": final_match,
            "before_rerank_match": before_match,
            "after_rerank_match": after_rerank_match,
        },
        ["候选阶段曾命中但最终未命中；先复查 final top-k 组装，再检查排序与去重逻辑。"],
    )
