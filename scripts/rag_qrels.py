from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

from backend.filename_normalization import normalize_filename_for_match
from scripts.rag_dataset_utils import as_list, load_jsonl, loose_text


QREL_SCHEMA_VERSION = "rag-chunk-gold-v2"
DEFAULT_ROOT_TYPE = "section"
DEFAULT_ROOT_GRANULARITY = "section_path"
VALID_QREL_CONFLICT_POLICIES = {"external", "dataset", "fail"}
VALID_QREL_MATCH_MODES = {"strict_id", "canonical"}


def normalized_text_hash(value: object, length: int = 16) -> str:
    normalized = loose_text(value)
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _page_bounds(chunk: dict[str, Any]) -> tuple[int, int]:
    page_number = _to_int(chunk.get("page_number"))
    page_start = _to_int(chunk.get("page_start"), page_number)
    page_end = _to_int(chunk.get("page_end"), page_number or page_start)
    if page_end and page_start and page_end < page_start:
        page_end = page_start
    return page_start, page_end


def canonical_chunk_id(chunk: dict[str, Any]) -> str:
    filename = normalize_filename_for_match(chunk.get("filename") or chunk.get("file_name") or "")
    page_start, page_end = _page_bounds(chunk)
    text = chunk.get("retrieval_text") or chunk.get("text") or chunk.get("text_preview") or ""
    text_hash = normalized_text_hash(text)
    payload = f"{filename}|{page_start}|{page_end}|{text_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_root_id(chunk: dict[str, Any], root_type: str = DEFAULT_ROOT_TYPE) -> str:
    filename = normalize_filename_for_match(chunk.get("filename") or chunk.get("file_name") or "")
    page_start, page_end = _page_bounds(chunk)
    section_path = str(chunk.get("section_path") or chunk.get("section_title") or "")
    anchor_id = str(chunk.get("anchor_id") or "")
    root_key = str(chunk.get("root_chunk_id") or chunk.get("parent_chunk_id") or "")
    payload = "|".join(
        [
            filename,
            root_type,
            loose_text(section_path),
            loose_text(anchor_id),
            f"{page_start}-{page_end}",
            root_key,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def attach_canonical_ids(chunk: dict[str, Any], root_type: str = DEFAULT_ROOT_TYPE) -> dict[str, Any]:
    out = dict(chunk)
    out["canonical_chunk_id"] = str(out.get("canonical_chunk_id") or canonical_chunk_id(out))
    out["canonical_root_id"] = str(out.get("canonical_root_id") or canonical_root_id(out, root_type=root_type))
    out.setdefault("root_type", root_type)
    out.setdefault("root_granularity", DEFAULT_ROOT_GRANULARITY)
    return out


def qrel_record_key(record: dict[str, Any]) -> str:
    for key in ("id", "sample_id", "query_id"):
        value = record.get(key)
        if value:
            return str(value)
    query = record.get("question") or record.get("query") or record.get("input")
    return normalized_text_hash(query, length=24)


def _unique_strings(values: object) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in as_list(values):
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _qrel_ids(record: dict[str, Any], key: str) -> list[str]:
    return _unique_strings(record.get(key))


def _qrel_field_differs(left: list[str], right: list[str]) -> bool:
    return bool(left and right and set(left) != set(right))


def _merge_field(
    base: dict[str, Any],
    external: dict[str, Any],
    key: str,
    *,
    policy: str,
    warnings: list[dict[str, Any]],
    sample_key: str,
) -> None:
    base_values = _qrel_ids(base, key)
    external_values = _qrel_ids(external, key)
    if not external_values:
        return
    if _qrel_field_differs(base_values, external_values):
        warning = {
            "sample_key": sample_key,
            "field": key,
            "dataset_values": base_values,
            "external_values": external_values,
        }
        warnings.append(warning)
        if policy == "fail":
            raise ValueError(f"qrel conflict for {sample_key} field={key}: {warning}")
    if policy == "dataset" and base_values:
        return
    base[key] = external_values


def _merge_quality(base: dict[str, Any], external: dict[str, Any], policy: str) -> None:
    external_quality = external.get("quality")
    if not isinstance(external_quality, dict):
        return
    if policy == "dataset" and isinstance(base.get("quality"), dict):
        return
    base["quality"] = dict(external_quality)


def load_chunk_qrels(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def merge_chunk_qrels(
    records: list[dict[str, Any]],
    qrel_rows: list[dict[str, Any]],
    *,
    conflict_policy: str = "external",
    chunk_match_mode: str = "strict_id",
    root_match_mode: str = "strict_id",
    qrel_source: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if conflict_policy not in VALID_QREL_CONFLICT_POLICIES:
        raise ValueError(f"invalid qrel conflict policy: {conflict_policy}")
    if chunk_match_mode not in VALID_QREL_MATCH_MODES:
        raise ValueError(f"invalid chunk qrel match mode: {chunk_match_mode}")
    if root_match_mode not in VALID_QREL_MATCH_MODES:
        raise ValueError(f"invalid root qrel match mode: {root_match_mode}")

    qrels_by_key = {qrel_record_key(row): row for row in qrel_rows}
    merged: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    matched = 0

    for record in records:
        out = dict(record)
        sample_key = qrel_record_key(record)
        qrel = qrels_by_key.get(sample_key)
        if qrel:
            matched += 1
            for field in (
                "gold_chunk_ids",
                "expected_chunk_ids",
                "legacy_gold_chunk_ids",
                "expected_root_ids",
                "canonical_chunk_ids",
                "expected_canonical_chunk_ids",
                "canonical_root_ids",
                "expected_canonical_root_ids",
            ):
                _merge_field(
                    out,
                    qrel,
                    field,
                    policy=conflict_policy,
                    warnings=warnings,
                    sample_key=sample_key,
                )
            for field in ("supporting_chunks", "root_type", "root_granularity", "index_profile", "collection"):
                if qrel.get(field) and not (conflict_policy == "dataset" and out.get(field)):
                    out[field] = qrel.get(field)
            _merge_quality(out, qrel, conflict_policy)
            out["qrel_source"] = qrel_source or "external"
        out["chunk_qrel_match_mode"] = chunk_match_mode
        out["root_qrel_match_mode"] = root_match_mode
        merged.append(out)

    review_counts = Counter()
    status_counts = Counter()
    for row in qrel_rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        review_counts[str(quality.get("review_status") or "missing")] += 1
        status_counts[str(quality.get("alignment_status") or "missing")] += 1

    report = {
        "type": "external_chunk_qrels",
        "schema_version": QREL_SCHEMA_VERSION,
        "qrel_source": qrel_source or "",
        "qrel_rows": len(qrel_rows),
        "dataset_rows": len(records),
        "matched_rows": matched,
        "unmatched_qrel_rows": max(0, len(qrel_rows) - matched),
        "chunk_qrel_match_mode": chunk_match_mode,
        "root_qrel_match_mode": root_match_mode,
        "qrel_conflict_policy": conflict_policy,
        "conflict_count": len(warnings),
        "conflicts": warnings[:25],
        "review_status_counts": dict(review_counts),
        "alignment_status_counts": dict(status_counts),
        "review_coverage": (
            (review_counts.get("approved", 0) + review_counts.get("rejected", 0)) / len(qrel_rows)
            if qrel_rows
            else 0.0
        ),
    }
    return merged, report


def qrel_ids_for_match(record: dict[str, Any], kind: str, mode: str) -> list[str]:
    if kind == "chunk":
        strict = _qrel_ids(record, "expected_chunk_ids") or _qrel_ids(record, "legacy_gold_chunk_ids") or _qrel_ids(record, "gold_chunk_ids")
        canonical = _qrel_ids(record, "expected_canonical_chunk_ids") or _qrel_ids(record, "canonical_chunk_ids")
    elif kind == "root":
        strict = _qrel_ids(record, "expected_root_ids")
        canonical = _qrel_ids(record, "expected_canonical_root_ids") or _qrel_ids(record, "canonical_root_ids")
    else:
        raise ValueError(f"unknown qrel kind: {kind}")
    return canonical if mode == "canonical" and canonical else strict


def qrel_report_output_paths(base: Path) -> dict[str, Path]:
    return {
        "output": base.with_suffix(".jsonl"),
        "review": base.with_name(f"{base.stem}.review.jsonl"),
        "report": base.with_name(f"{base.stem}.report.json"),
        "failed_sample": base.with_name(f"{base.stem}.failed.sample.jsonl"),
        "ambiguous_sample": base.with_name(f"{base.stem}.ambiguous.sample.jsonl"),
    }


def page_distance(row: dict[str, Any], chunk: dict[str, Any]) -> int | None:
    pages = []
    for value in as_list(row.get("gold_pages")) or as_list(row.get("expected_pages")):
        if re.fullmatch(r"\d+", str(value)):
            pages.append(int(value))
    if not pages:
        return None
    page_start, page_end = _page_bounds(chunk)
    if page_start <= min(pages) <= page_end or page_start <= max(pages) <= page_end:
        return 0
    return min(abs(page_start - page) for page in pages)
