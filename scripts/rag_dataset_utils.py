from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from backend.shared.filename_normalization import normalize_filename_for_match


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "rag_doc_gold.jsonl"


@dataclass(frozen=True)
class AlignmentScore:
    score: float
    method: str
    reasons: tuple[str, ...]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_digest(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def loose_text(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(value or "")).lower()


def as_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def record_source_file(row: dict[str, Any]) -> str:
    gold_files = as_list(row.get("gold_files"))
    if gold_files:
        return str(gold_files[0])
    metadata = row.get("metadata") or {}
    return str(metadata.get("source_file") or row.get("source_file") or "")


def group_rows_by_source_file(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[record_source_file(row)].append(row)
    return dict(grouped)


def _row_pages(row: dict[str, Any]) -> set[int]:
    pages = {int(page) for page in as_list(row.get("gold_pages")) if str(page).isdigit()}
    if not pages:
        pages = {int(page) for page in as_list(row.get("expected_pages")) if str(page).isdigit()}
    return pages


def chunk_page_distance(row: dict[str, Any], chunk: dict[str, Any]) -> int | None:
    pages = _row_pages(row)
    if not pages:
        return None
    page_number = int(chunk.get("page_number") or 0)
    page_start = int(chunk.get("page_start") or page_number or 0)
    page_end = int(chunk.get("page_end") or page_number or 0)
    if page_number in pages or any(page_start <= page <= page_end for page in pages):
        return 0
    candidates = [value for value in (page_number, page_start, page_end) if value]
    if not candidates:
        return None
    return min(abs(candidate - page) for candidate in candidates for page in pages)


def chunk_page_matches(row: dict[str, Any], chunk: dict[str, Any]) -> bool:
    distance = chunk_page_distance(row, chunk)
    return distance is None or distance == 0


def chunk_file_matches(row: dict[str, Any], chunk: dict[str, Any]) -> bool:
    return normalize_filename_for_match(record_source_file(row)) == normalize_filename_for_match(
        chunk.get("filename") or chunk.get("file_name") or ""
    )


def alignment_score(row: dict[str, Any], chunk: dict[str, Any]) -> AlignmentScore:
    reasons: list[str] = []
    score = 0.0
    text = str(chunk.get("text") or "")
    retrieval_text = str(chunk.get("retrieval_text") or "")
    combined = "\n".join(
        [
            text,
            retrieval_text,
            str(chunk.get("section_title") or ""),
            str(chunk.get("section_path") or ""),
        ]
    )
    loose_combined = loose_text(combined)
    loose_excerpt = loose_text(row.get("source_excerpt") or row.get("expected_answer") or "")

    if chunk_file_matches(row, chunk):
        score += 0.2
        reasons.append("file")
    page_distance = chunk_page_distance(row, chunk)
    if page_distance is None:
        score += 0.2
        reasons.append("page_unscored")
    elif page_distance == 0:
        score += 0.2
        reasons.append("page")
    elif page_distance == 1:
        score += 0.05
        reasons.append("neighbor_page")
    if loose_excerpt and loose_excerpt in loose_combined:
        score += 0.45
        reasons.append("excerpt")
    elif loose_excerpt:
        windows = [
            loose_excerpt[i : i + 20]
            for i in range(0, len(loose_excerpt), 20)
            if len(loose_excerpt[i : i + 20]) >= 8
        ]
        if windows:
            coverage = sum(1 for window in windows if window in loose_combined) / len(windows)
            score += 0.25 * coverage
            if coverage:
                reasons.append("window")

    anchors = [str(item) for item in as_list(row.get("expected_anchors")) if str(item)]
    anchor_text = " ".join(
        [
            str(chunk.get("anchor_id") or ""),
            str(chunk.get("section_title") or ""),
            str(chunk.get("section_path") or ""),
            text,
        ]
    )
    if anchors and any(anchor and anchor in anchor_text for anchor in anchors):
        score += 0.1
        reasons.append("anchor")

    keywords = [loose_text(item) for item in as_list(row.get("expected_keywords")) if loose_text(item)]
    if keywords:
        hits = sum(1 for keyword in keywords if keyword in loose_combined)
        if hits:
            score += min(0.05, 0.05 * hits / max(1, min(3, len(keywords))))
            reasons.append("keywords")

    source_headings = [
        loose_text(item)
        for item in (
            [row.get("source_heading")]
            + [ctx.get("heading") for ctx in as_list(row.get("positive_contexts")) if isinstance(ctx, dict)]
        )
        if loose_text(item)
    ]
    section_text = loose_text(" ".join([str(chunk.get("section_title") or ""), str(chunk.get("section_path") or "")]))
    if source_headings and section_text:
        if any(heading in section_text or section_text in heading for heading in source_headings):
            score += 0.05
            reasons.append("same_section")

    method = "none"
    if "excerpt" in reasons:
        method = "exact"
    elif "window" in reasons:
        method = "window"
    elif "anchor" in reasons or "keywords" in reasons or "same_section" in reasons:
        method = "anchor_keyword"

    return AlignmentScore(score=min(score, 1.0), method=method, reasons=tuple(reasons))
