from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.shared.filename_normalization import normalize_filename_for_match, raw_filename_basename  # noqa: E402
from scripts.rag_eval.variants import (  # noqa: E402
    DATASET_DIR,
    DEFAULT_CANONICAL_CORPUS,
    DEFAULT_FROZEN_DATASET,
    DEFAULT_GOLD_DATASET,
    DEFAULT_S3_COLLECTION,
    DEFAULT_VARIANTS,
    EVAL_SCHEMA_VERSION,
    PAIR_DEFINITIONS,
    VARIANT_CONFIGS,
)
from scripts.rag_qrels import (  # noqa: E402
    VALID_QREL_CONFLICT_POLICIES,
    VALID_QREL_MATCH_MODES,
    attach_canonical_ids,
    load_chunk_qrels,
    merge_chunk_qrels,
)
from scripts.rag_eval.metrics import (  # noqa: E402
    compare_sample_rank as _compare_sample_rank,
    summarize_results as _summarize_results,
)
from scripts.rag_eval.io import load_jsonl as _load_jsonl, write_jsonl as _write_jsonl  # noqa: E402
from scripts.rag_eval.common import (  # noqa: E402
    as_list as _common_as_list,
    doc_filename_norm as _common_doc_filename_norm,
    normalized_filename_set as _common_normalized_filename_set,
    percentile_values as _common_percentile_values,
    present as _common_present,
    rate as _common_rate,
)
from scripts.rag_eval.preflight import (  # noqa: E402
    validate_eval_dataset_records as _validate_eval_dataset_records,
)
from scripts.rag_eval.reporting import render_summary_markdown as _render_summary_markdown  # noqa: E402
from scripts.rag_eval.regression import compare_core_summary  # noqa: E402

_QRELS_NA = "n/a (qrels missing)"




def _as_list(value: Any) -> list[str]:
    return _common_as_list(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_dataset_profile(profile: str) -> Path:
    if profile == "frozen":
        return DEFAULT_FROZEN_DATASET
    if profile == "gold":
        return DEFAULT_GOLD_DATASET
    if profile == "smoke":
        return DEFAULT_FROZEN_DATASET
    if profile == "natural":
        natural_path = DATASET_DIR / "rag_doc_gold_natural_v1.jsonl"
        if natural_path.is_file():
            return natural_path
        raise FileNotFoundError(
            f"Natural dataset not found at {natural_path}. "
            "Run scripts/derive_natural_query_subset.py first."
        )
    raise ValueError("--dataset is required when --dataset-profile custom is used")


def _dataset_schema_versions(records: list[dict]) -> list[str]:
    versions = {
        str(record.get("benchmark_schema_version") or "")
        for record in records
        if record.get("benchmark_schema_version")
    }
    return sorted(versions)


def validate_eval_dataset_records(records: list[dict], dataset: Path) -> dict[str, Any]:
    return _validate_eval_dataset_records(records, dataset=str(dataset))


def _expected_files_from_records(records: list[dict]) -> set[str]:
    expected: set[str] = set()
    for record in records:
        expected.update(_as_list(record.get("expected_files")) or _as_list(record.get("gold_files")))
    return expected


def _normalized_filename_set(values: Any) -> set[str]:
    return _common_normalized_filename_set(values)


def _doc_filename_norm(doc: dict) -> str:
    return _common_doc_filename_norm(doc)


def _file_coverage_details(expected_files: set[str], indexed_files: set[str]) -> dict[str, Any]:
    expected_raw = sorted({raw_filename_basename(item) for item in expected_files if raw_filename_basename(item)})
    indexed_raw = sorted({raw_filename_basename(item) for item in indexed_files if raw_filename_basename(item)})
    indexed_raw_set = set(indexed_raw)

    indexed_by_norm: dict[str, list[str]] = defaultdict(list)
    for raw in indexed_raw:
        norm = normalize_filename_for_match(raw)
        if norm:
            indexed_by_norm[norm].append(raw)

    matches = []
    covered = 0
    exact_covered = 0
    missing: list[str] = []
    for raw in expected_raw:
        norm = normalize_filename_for_match(raw)
        exact_match = raw in indexed_raw_set
        normalized_matches = indexed_by_norm.get(norm, [])
        matched = exact_match or bool(normalized_matches)
        if matched:
            covered += 1
        else:
            missing.append(raw)
        if exact_match:
            exact_covered += 1
        matches.append(
            {
                "raw_expected": raw,
                "normalized_expected": norm,
                "raw_indexed": raw if exact_match else (normalized_matches[0] if normalized_matches else None),
                "normalized_indexed": norm if normalized_matches else None,
                "match_method": "exact" if exact_match else ("normalized" if normalized_matches else "missing"),
            }
        )

    expected_count = len(expected_raw)
    return {
        "expected_unique_files": expected_count,
        "indexed_unique_files": len(indexed_raw),
        "covered_files": covered,
        "exact_covered_files": exact_covered,
        "coverage": (covered / expected_count) if expected_count else 1.0,
        "exact_coverage": (exact_covered / expected_count) if expected_count else 1.0,
        "missing_files": missing,
        "file_matches": matches,
    }


def _corpus_coverage_report(records: list[dict], documents_dir: Path) -> dict[str, Any]:
    expected = _expected_files_from_records(records)
    corpus_files = {path.name for path in documents_dir.iterdir() if path.is_file()} if documents_dir.is_dir() else set()
    details = _file_coverage_details(expected, corpus_files)
    return {
        "type": "corpus_file_coverage",
        "documents_dir": str(documents_dir),
        "corpus_files": len(corpus_files),
        **details,
    }


def _rate(count: int, total: int) -> float:
    return _common_rate(count, total)


def _percentile_values(values: list[float], percentile: float) -> float | None:
    return _common_percentile_values(values, percentile)


def _present(value: Any) -> bool:
    return _common_present(value)


def _metadata_coverage_report(rows: list[dict]) -> dict[str, Any]:
    total = len(rows)
    retrieval_lengths = [float(len(str(row.get("retrieval_text") or ""))) for row in rows]
    filename_rate = _rate(sum(1 for row in rows if _present(row.get("filename"))), total)
    page_rate = _rate(sum(1 for row in rows if _present(row.get("page_number")) or _present(row.get("page_start"))), total)
    retrieval_rate = _rate(sum(1 for length in retrieval_lengths if length > 0), total)
    retrieval_p95 = _percentile_values(retrieval_lengths, 0.95)

    report = {
        "leaf_rows": total,
        "filename_non_empty_rate": filename_rate,
        "page_number_or_page_start_rate": page_rate,
        "retrieval_text_non_empty_rate": retrieval_rate,
        "section_path_non_empty_rate": _rate(sum(1 for row in rows if _present(row.get("section_path"))), total),
        "anchor_id_non_empty_rate": _rate(sum(1 for row in rows if _present(row.get("anchor_id"))), total),
        "parent_chunk_id_non_empty_rate": _rate(sum(1 for row in rows if _present(row.get("parent_chunk_id"))), total),
        "root_chunk_id_non_empty_rate": _rate(sum(1 for row in rows if _present(row.get("root_chunk_id"))), total),
        "chunk_id_non_empty_rate": _rate(sum(1 for row in rows if _present(row.get("chunk_id"))), total),
        "retrieval_text_length_p50": _percentile_values(retrieval_lengths, 0.50),
        "retrieval_text_length_p95": retrieval_p95,
        "retrieval_text_length_max": max(retrieval_lengths) if retrieval_lengths else None,
    }
    failures = []
    hard_gates = {
        "filename_non_empty_rate": (filename_rate, 0.99, ">="),
        "page_number_or_page_start_rate": (page_rate, 0.95, ">="),
        "retrieval_text_non_empty_rate": (retrieval_rate, 0.99, ">="),
        "retrieval_text_length_p95": (retrieval_p95, 4000.0, "<="),
    }
    for metric, (value, threshold, op) in hard_gates.items():
        ok = value is not None and (value >= threshold if op == ">=" else value <= threshold)
        if not ok:
            failures.append({"metric": metric, "value": value, "threshold": threshold, "operator": op})
    report["hard_gate_failures"] = failures
    report["hard_gate_pass"] = not failures
    return report


def _collection_coverage_report(records: list[dict], variant: str) -> dict[str, Any]:
    expected = _expected_files_from_records(records)
    env = _merged_env(variant)
    old_values = {key: os.environ.get(key) for key in ("MILVUS_COLLECTION", "RAG_INDEX_PROFILE")}
    os.environ["MILVUS_COLLECTION"] = env.get("MILVUS_COLLECTION") or "embeddings_collection"
    os.environ["RAG_INDEX_PROFILE"] = env.get("RAG_INDEX_PROFILE") or "legacy"
    try:
        from backend.infra.vector_store.milvus_client import MilvusManager

        manager = MilvusManager()
        rows = manager.query_all(
            filter_expr="chunk_level == 3",
            output_fields=[
                "filename",
                "page_number",
                "page_start",
                "retrieval_text",
                "section_path",
                "anchor_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_id",
            ],
        )
        indexed = {str(row.get("filename") or "") for row in rows}
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    details = _file_coverage_details(expected, indexed)
    return {
        "type": "collection_closure",
        "variant": variant,
        "collection": env.get("MILVUS_COLLECTION") or "embeddings_collection",
        "index_profile": env.get("RAG_INDEX_PROFILE") or "legacy",
        **details,
        "metadata_coverage": _metadata_coverage_report(rows),
    }


def _qrel_ids_from_record(record: dict) -> set[str]:
    judgment_ids = set(_relevance_judgment_map(record.get("relevance_judgments")).keys())
    if judgment_ids:
        return judgment_ids
    context_ids = _qrel_ids_from_contexts(record.get("positive_contexts"))
    if context_ids:
        return context_ids
    page_refs = set(_as_list(record.get("expected_page_refs")))
    if page_refs:
        return page_refs
    return set(_as_list(record.get("gold_doc_ids")))


def _qrel_coverage_report(records: list[dict]) -> dict[str, Any]:
    total = len(records)
    file_count = sum(1 for record in records if _as_list(record.get("expected_files")) or _as_list(record.get("gold_files")))
    page_count = sum(1 for record in records if _qrel_ids_from_record(record))
    chunk_count = sum(
        1
        for record in records
        if _as_list(record.get("expected_chunk_ids"))
        or _as_list(record.get("legacy_gold_chunk_ids"))
        or _as_list(record.get("gold_chunk_ids"))
        or _as_list(record.get("expected_canonical_chunk_ids"))
        or _as_list(record.get("canonical_chunk_ids"))
    )
    root_count = sum(
        1
        for record in records
        if _as_list(record.get("expected_root_ids"))
        or _as_list(record.get("expected_canonical_root_ids"))
        or _as_list(record.get("canonical_root_ids"))
    )
    return {
        "type": "qrel_coverage",
        "rows": total,
        "total_rows": total,
        "file_qrel_samples": file_count,
        "page_qrel_samples": page_count,
        "chunk_qrel_samples": chunk_count,
        "root_qrel_samples": root_count,
        "chunk_qrel_rows": chunk_count,
        "root_qrel_rows": root_count,
        "file_qrel_coverage": _rate(file_count, total),
        "page_qrel_coverage": _rate(page_count, total),
        "chunk_qrel_coverage": _rate(chunk_count, total),
        "root_qrel_coverage": _rate(root_count, total),
    }


def _doc_text(doc: dict) -> str:
    parts = [
        doc.get("anchor_id"),
        doc.get("section_title"),
        doc.get("section_path"),
        doc.get("retrieval_text"),
        doc.get("text"),
        doc.get("text_preview"),
    ]
    return " ".join(str(part) for part in parts if part)


def _doc_page_candidates(doc: dict) -> set[str]:
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
        # Keep the candidate set bounded; long ranges usually indicate bad metadata.
        for page in range(page_start, min(page_end, page_start + 20) + 1):
            pages.add(str(page))
            pages.add(str(page + 1))
    elif page_number is None and page_start is None and page_end is None:
        return pages

    return pages


def _exact_page_candidates(doc: dict) -> set[str]:
    pages: set[str] = set()
    for key in ("page_number", "page_start", "page_end"):
        try:
            page = int(doc.get(key))
        except (TypeError, ValueError):
            continue
        pages.add(str(page))
    return pages


def _page_ref_aliases(filename: str, page: str) -> set[str]:
    raw = raw_filename_basename(filename)
    norm = normalize_filename_for_match(filename)
    refs = {f"{raw}::p{page}"} if raw else set()
    if norm:
        refs.add(f"{norm}::p{page}")
    return refs


def _qrel_id_aliases(qrel_id: str) -> set[str]:
    text = str(qrel_id or "")
    aliases = {text} if text else set()
    match = re.match(r"(.+)::p(\d+)$", text)
    if match:
        aliases |= _page_ref_aliases(match.group(1), match.group(2))
    return aliases


def _canonical_qrel_id(qrel_id: str) -> str:
    text = str(qrel_id or "")
    match = re.match(r"(.+)::p(\d+)$", text)
    if not match:
        return text
    return f"{normalize_filename_for_match(match.group(1))}::p{int(match.group(2))}"


def _page_ref_aliases_from_values(values: list[str] | None) -> set[str]:
    aliases: set[str] = set()
    for value in _as_list(values):
        aliases |= _qrel_id_aliases(value)
    return aliases


def _doc_page_refs(doc: dict) -> set[str]:
    filename = str(doc.get("filename") or "")
    refs: set[str] = set()
    for page in _doc_page_candidates(doc):
        refs |= _page_ref_aliases(filename, page)
    return refs


def _doc_qrel_ids(doc: dict) -> set[str]:
    ids = {str(doc.get("chunk_id") or ""), str(doc.get("root_chunk_id") or "")}
    ids |= _doc_page_refs(doc)
    return {item for item in ids if item}


def _doc_with_canonical_ids(doc: dict) -> dict[str, Any]:
    if doc.get("canonical_chunk_id") and doc.get("canonical_root_id"):
        return doc
    try:
        return attach_canonical_ids(doc)
    except Exception:
        return dict(doc)


def _doc_chunk_ids_for_mode(doc: dict, mode: str) -> set[str]:
    if mode == "canonical":
        canonical_doc = _doc_with_canonical_ids(doc)
        return {str(canonical_doc.get("canonical_chunk_id") or "")} - {""}
    return {str(doc.get("chunk_id") or "")} - {""}


def _doc_root_ids_for_mode(doc: dict, mode: str) -> set[str]:
    if mode == "canonical":
        canonical_doc = _doc_with_canonical_ids(doc)
        return {str(canonical_doc.get("canonical_root_id") or "")} - {""}
    return {str(doc.get("root_chunk_id") or "")} - {""}


def _qrel_ids_from_contexts(contexts: Any) -> set[str]:
    ids: set[str] = set()
    for item in contexts or []:
        if not isinstance(item, dict):
            continue
        for key in ("doc_id", "chunk_id", "root_chunk_id"):
            value = item.get(key)
            if value:
                ids |= _qrel_id_aliases(str(value))
        file_name = item.get("file_name") or item.get("filename")
        if file_name:
            for page in _exact_page_candidates(item):
                ids |= _page_ref_aliases(str(file_name), page)
    return ids


def _canonical_qrel_ids(
    positive_contexts: Any,
    relevance_judgments: Any,
    expected_page_refs: list[str] | None,
) -> set[str]:
    judgment_ids = {
        _canonical_qrel_id(str(item.get("doc_id") or item.get("chunk_id")))
        for item in relevance_judgments or []
        if isinstance(item, dict) and (item.get("doc_id") or item.get("chunk_id"))
    }
    if judgment_ids:
        return judgment_ids

    context_ids: set[str] = set()
    for item in positive_contexts or []:
        if not isinstance(item, dict):
            continue
        for key in ("doc_id", "chunk_id", "root_chunk_id"):
            value = item.get(key)
            if value:
                context_ids.add(_canonical_qrel_id(str(value)))
        file_name = item.get("file_name") or item.get("filename")
        if file_name:
            for page in _exact_page_candidates(item):
                context_ids.add(_canonical_qrel_id(f"{file_name}::p{page}"))
    if context_ids:
        return context_ids

    return {_canonical_qrel_id(item) for item in _as_list(expected_page_refs)}


def _relevance_judgment_map(judgments: Any) -> dict[str, float]:
    mapped: dict[str, float] = {}
    for item in judgments or []:
        if not isinstance(item, dict):
            continue
        doc_id = item.get("doc_id") or item.get("chunk_id")
        if not doc_id:
            continue
        try:
            relevance = float(item.get("relevance", 1.0))
        except (TypeError, ValueError):
            relevance = 1.0
        for alias in _qrel_id_aliases(str(doc_id)):
            mapped[alias] = max(mapped.get(alias, 0.0), relevance)
    return mapped


def _doc_relevance(doc: dict, qrels: dict[str, float], seen_qrels: set[str] | None = None) -> tuple[float, str | None]:
    best_id = None
    best_score = 0.0
    for qrel_id in _doc_qrel_ids(doc):
        canonical_id = _canonical_qrel_id(qrel_id)
        if seen_qrels is not None and canonical_id in seen_qrels:
            continue
        score = max(qrels.get(qrel_id, 0.0), qrels.get(canonical_id, 0.0))
        if score > best_score:
            best_id = canonical_id
            best_score = score
    return best_score, best_id


def _canonical_relevance_values(qrels: dict[str, float]) -> list[float]:
    canonical: dict[str, float] = {}
    for qrel_id, relevance in qrels.items():
        canonical_id = _canonical_qrel_id(qrel_id)
        canonical[canonical_id] = max(canonical.get(canonical_id, 0.0), relevance)
    return sorted(canonical.values(), reverse=True)


def _discounted_cumulative_gain(relevances: list[float]) -> float:
    return sum(((2.0**rel - 1.0) / math.log2(idx + 2)) for idx, rel in enumerate(relevances))


_NUMERAL_NE = r"(?<![一二三四五六七八九十百千万零两\d])"
_NUMERAL_NLA = r"(?![一二三四五六七八九十百千万零两\d])"


def _anchor_match(anchor: str, text: str) -> bool:
    """Match anchor as a discrete structural unit, not a partial substring.

    Prevents ``"1.2"`` matching ``"11.2"``, ``"一、"`` matching ``"二十一、"``,
    and similar false positives for decimal/list-item anchors.
    """
    if not anchor or not text:
        return False
    escaped = re.escape(anchor)
    return bool(re.search(_NUMERAL_NE + escaped + _NUMERAL_NLA, text))


def _doc_match_flags(
    doc: dict,
    expected_chunk_ids: list[str] | None = None,
    expected_root_ids: list[str] | None = None,
    expected_canonical_chunk_ids: list[str] | None = None,
    expected_canonical_root_ids: list[str] | None = None,
    chunk_qrel_match_mode: str = "strict_id",
    root_qrel_match_mode: str = "strict_id",
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    expected_files: list[str] | None = None,
    expected_pages: list[str] | None = None,
    expected_page_refs: list[str] | None = None,
    expected_all_keywords: list[str] | None = None,
    expected_keyword_policy: dict[str, Any] | None = None,
) -> dict[str, bool]:
    chunk_mode = chunk_qrel_match_mode if chunk_qrel_match_mode in VALID_QREL_MATCH_MODES else "strict_id"
    root_mode = root_qrel_match_mode if root_qrel_match_mode in VALID_QREL_MATCH_MODES else "strict_id"
    chunk_ids = (
        set(_as_list(expected_canonical_chunk_ids))
        if chunk_mode == "canonical" and _as_list(expected_canonical_chunk_ids)
        else set(_as_list(expected_chunk_ids))
    )
    root_ids = (
        set(_as_list(expected_canonical_root_ids))
        if root_mode == "canonical" and _as_list(expected_canonical_root_ids)
        else set(_as_list(expected_root_ids))
    )
    anchors = _as_list(expected_anchors)
    keywords = _as_list(expected_keywords)
    all_keywords = _as_list(expected_all_keywords) or keywords
    files = _normalized_filename_set(expected_files)
    pages = set(_as_list(expected_pages))
    page_refs = _page_ref_aliases_from_values(expected_page_refs)
    text = _doc_text(doc)

    chunk_hit = bool(chunk_ids and (_doc_chunk_ids_for_mode(doc, chunk_mode) & chunk_ids))
    root_hit = bool(root_ids and (_doc_root_ids_for_mode(doc, root_mode) & root_ids))
    anchor_hit = any(_anchor_match(anchor, text) for anchor in anchors)
    keyword_hit = any(keyword and keyword in text for keyword in keywords)
    keyword_matches = sum(1 for keyword in all_keywords if keyword and keyword in text)
    min_keyword_match = int((expected_keyword_policy or {}).get("min_match") or (1 if all_keywords else 0))
    keyword_required_hit = bool(all_keywords) and keyword_matches >= min_keyword_match
    file_hit = bool(_doc_filename_norm(doc) in files)
    page_hit = bool(pages and _doc_page_candidates(doc) & pages)
    page_ref_hit = bool(page_refs and _doc_page_refs(doc) & page_refs)

    return {
        "chunk": chunk_hit,
        "root": root_hit,
        "anchor": anchor_hit,
        "keyword": keyword_hit,
        "keyword_required": keyword_required_hit,
        "file": file_hit,
        "page": page_hit or page_ref_hit,
        "any": chunk_hit or root_hit or anchor_hit or keyword_required_hit or keyword_hit or page_ref_hit or page_hit or file_hit,
    }


def first_relevant_rank(
    docs: list[dict],
    expected_chunk_ids: list[str] | None = None,
    expected_root_ids: list[str] | None = None,
    expected_canonical_chunk_ids: list[str] | None = None,
    expected_canonical_root_ids: list[str] | None = None,
    chunk_qrel_match_mode: str = "strict_id",
    root_qrel_match_mode: str = "strict_id",
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    expected_files: list[str] | None = None,
    expected_pages: list[str] | None = None,
    expected_page_refs: list[str] | None = None,
    expected_all_keywords: list[str] | None = None,
    expected_keyword_policy: dict[str, Any] | None = None,
    top_k: int = 5,
) -> int | None:
    prefer_chunk = bool(_as_list(expected_chunk_ids) or _as_list(expected_canonical_chunk_ids))
    for idx, doc in enumerate((docs or [])[:top_k], 1):
        flags = _doc_match_flags(
            doc,
            expected_chunk_ids=expected_chunk_ids,
            expected_root_ids=expected_root_ids,
            expected_canonical_chunk_ids=expected_canonical_chunk_ids,
            expected_canonical_root_ids=expected_canonical_root_ids,
            chunk_qrel_match_mode=chunk_qrel_match_mode,
            root_qrel_match_mode=root_qrel_match_mode,
            expected_anchors=expected_anchors,
            expected_keywords=expected_keywords,
            expected_files=expected_files,
            expected_pages=expected_pages,
            expected_page_refs=expected_page_refs,
            expected_all_keywords=expected_all_keywords,
            expected_keyword_policy=expected_keyword_policy,
        )
        if prefer_chunk and flags["chunk"]:
            return idx
        if not prefer_chunk and flags["any"]:
            return idx
    return None


def _count_expected_matches(
    top_docs: list[dict],
    expected_chunk_ids: list[str] | None = None,
    expected_root_ids: list[str] | None = None,
    expected_canonical_chunk_ids: list[str] | None = None,
    expected_canonical_root_ids: list[str] | None = None,
    chunk_qrel_match_mode: str = "strict_id",
    root_qrel_match_mode: str = "strict_id",
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    expected_files: list[str] | None = None,
    expected_pages: list[str] | None = None,
    expected_page_refs: list[str] | None = None,
    expected_all_keywords: list[str] | None = None,
    expected_keyword_policy: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Return (matched_count, total_count) for distinct expected items found in top docs."""
    chunk_mode = chunk_qrel_match_mode if chunk_qrel_match_mode in VALID_QREL_MATCH_MODES else "strict_id"
    root_mode = root_qrel_match_mode if root_qrel_match_mode in VALID_QREL_MATCH_MODES else "strict_id"
    chunk_ids = (
        set(_as_list(expected_canonical_chunk_ids))
        if chunk_mode == "canonical" and _as_list(expected_canonical_chunk_ids)
        else set(_as_list(expected_chunk_ids))
    )
    root_ids = (
        set(_as_list(expected_canonical_root_ids))
        if root_mode == "canonical" and _as_list(expected_canonical_root_ids)
        else set(_as_list(expected_root_ids))
    )
    anchors = _as_list(expected_anchors)
    keywords = _as_list(expected_keywords)
    files = _normalized_filename_set(expected_files)
    pages = set(_as_list(expected_pages))
    page_refs = {_canonical_qrel_id(page_ref) for page_ref in _as_list(expected_page_refs)}

    total = len(chunk_ids) + len(root_ids) + len(anchors) + len(keywords) + len(files) + len(pages) + len(page_refs)
    if total == 0:
        return 0, 0

    found = 0
    doc_chunk_ids = set().union(*(_doc_chunk_ids_for_mode(doc, chunk_mode) for doc in top_docs)) if top_docs else set()
    doc_root_ids = set().union(*(_doc_root_ids_for_mode(doc, root_mode) for doc in top_docs)) if top_docs else set()
    doc_files = {_doc_filename_norm(doc) for doc in top_docs}
    doc_pages = set().union(*(_doc_page_candidates(doc) for doc in top_docs)) if top_docs else set()
    doc_page_refs = set().union(*(_doc_page_refs(doc) for doc in top_docs)) if top_docs else set()
    found += len(chunk_ids & doc_chunk_ids)
    found += len(root_ids & doc_root_ids)
    found += len(files & doc_files)
    found += len(pages & doc_pages)
    found += sum(1 for page_ref in page_refs if _qrel_id_aliases(page_ref) & doc_page_refs)

    for anchor in anchors:
        if any(_anchor_match(anchor, _doc_text(doc)) for doc in top_docs):
            found += 1
    for keyword in keywords:
        if any(keyword and keyword in _doc_text(doc) for doc in top_docs):
            found += 1

    return found, total


def compute_retrieval_metrics(
    docs: list[dict],
    expected_chunk_ids: list[str] | None = None,
    expected_root_ids: list[str] | None = None,
    expected_canonical_chunk_ids: list[str] | None = None,
    expected_canonical_root_ids: list[str] | None = None,
    chunk_qrel_match_mode: str = "strict_id",
    root_qrel_match_mode: str = "strict_id",
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    expected_files: list[str] | None = None,
    expected_pages: list[str] | None = None,
    expected_page_refs: list[str] | None = None,
    expected_all_keywords: list[str] | None = None,
    expected_keyword_policy: dict[str, Any] | None = None,
    positive_contexts: list[dict] | None = None,
    relevance_judgments: list[dict] | None = None,
    hard_negative_files: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    top_docs = (docs or [])[:top_k]
    positive_qrel_ids = _qrel_ids_from_contexts(positive_contexts)
    judgment_qrels = _relevance_judgment_map(relevance_judgments)
    for qrel_id in positive_qrel_ids:
        judgment_qrels.setdefault(qrel_id, 1.0)
    if not positive_qrel_ids:
        for qrel_id in _as_list(expected_page_refs):
            positive_qrel_ids |= _qrel_id_aliases(qrel_id)
        for qrel_id in positive_qrel_ids:
            judgment_qrels.setdefault(qrel_id, 1.0)
    hard_negative_file_set = _normalized_filename_set(hard_negative_files)
    file_qrel_available = bool(_as_list(expected_files))
    page_qrel_available = bool(
        judgment_qrels
        or positive_qrel_ids
        or _as_list(expected_page_refs)
        or _as_list(expected_pages)
    )
    chunk_qrel_available = bool(_as_list(expected_chunk_ids) or _as_list(expected_canonical_chunk_ids))
    root_qrel_available = bool(_as_list(expected_root_ids) or _as_list(expected_canonical_root_ids))
    has_expected = bool(
        _as_list(expected_chunk_ids)
        or _as_list(expected_root_ids)
        or _as_list(expected_canonical_chunk_ids)
        or _as_list(expected_canonical_root_ids)
        or _as_list(expected_anchors)
        or _as_list(expected_keywords)
        or _as_list(expected_files)
        or _as_list(expected_pages)
        or _as_list(expected_page_refs)
        or positive_qrel_ids
    )
    flags = [
        _doc_match_flags(
            doc,
            expected_chunk_ids=expected_chunk_ids,
            expected_root_ids=expected_root_ids,
            expected_canonical_chunk_ids=expected_canonical_chunk_ids,
            expected_canonical_root_ids=expected_canonical_root_ids,
            chunk_qrel_match_mode=chunk_qrel_match_mode,
            root_qrel_match_mode=root_qrel_match_mode,
            expected_anchors=expected_anchors,
            expected_keywords=expected_keywords,
            expected_files=expected_files,
            expected_pages=expected_pages,
            expected_page_refs=expected_page_refs,
            expected_all_keywords=expected_all_keywords,
            expected_keyword_policy=expected_keyword_policy,
        )
        for doc in top_docs
    ]
    prefer_chunk = chunk_qrel_available
    relevant_count = sum(1 for item in flags if item["chunk"]) if prefer_chunk else sum(1 for item in flags if item["any"])
    rank = first_relevant_rank(
        top_docs,
        expected_chunk_ids=expected_chunk_ids,
        expected_root_ids=expected_root_ids,
        expected_canonical_chunk_ids=expected_canonical_chunk_ids,
        expected_canonical_root_ids=expected_canonical_root_ids,
        chunk_qrel_match_mode=chunk_qrel_match_mode,
        root_qrel_match_mode=root_qrel_match_mode,
        expected_anchors=expected_anchors,
        expected_keywords=expected_keywords,
        expected_files=expected_files,
        expected_pages=expected_pages,
        expected_page_refs=expected_page_refs,
        expected_all_keywords=expected_all_keywords,
        expected_keyword_policy=expected_keyword_policy,
        top_k=top_k,
    )
    returned_count = len(top_docs)
    precision = (relevant_count / returned_count) if returned_count else 0.0
    id_matches: set[str] = set()
    id_relevance_by_rank: list[float] = []
    average_precision_hits = 0
    average_precision_total = 0.0
    for rank_idx, doc in enumerate(top_docs, 1):
        rel, matched_id = _doc_relevance(doc, judgment_qrels, seen_qrels=id_matches)
        if matched_id:
            id_matches.add(matched_id)
        id_relevance_by_rank.append(rel)
        if rel > 0:
            average_precision_hits += 1
            average_precision_total += average_precision_hits / rank_idx

    positive_qrel_count = len(_canonical_qrel_ids(positive_contexts, relevance_judgments, expected_page_refs))
    if not positive_qrel_count:
        positive_qrel_count = len(judgment_qrels)
    id_precision = (len(id_matches) / returned_count) if returned_count and positive_qrel_count else None
    id_recall = (len(id_matches) / positive_qrel_count) if positive_qrel_count else None
    ideal_relevances = _canonical_relevance_values(judgment_qrels)[:top_k]
    ideal_dcg = _discounted_cumulative_gain(ideal_relevances)
    ndcg = (_discounted_cumulative_gain(id_relevance_by_rank) / ideal_dcg) if ideal_dcg else None
    map_at_k = (average_precision_total / positive_qrel_count) if positive_qrel_count else None
    hard_negative_count = sum(1 for doc in top_docs if _doc_filename_norm(doc) in hard_negative_file_set)
    hard_negative_ratio = (hard_negative_count / returned_count) if returned_count and hard_negative_file_set else None

    matched_expected, total_expected = _count_expected_matches(
        top_docs,
        expected_chunk_ids=expected_chunk_ids,
        expected_root_ids=expected_root_ids,
        expected_canonical_chunk_ids=expected_canonical_chunk_ids,
        expected_canonical_root_ids=expected_canonical_root_ids,
        chunk_qrel_match_mode=chunk_qrel_match_mode,
        root_qrel_match_mode=root_qrel_match_mode,
        expected_anchors=expected_anchors,
        expected_keywords=expected_keywords,
        expected_files=expected_files,
        expected_pages=expected_pages,
        expected_page_refs=expected_page_refs,
        expected_all_keywords=expected_all_keywords,
        expected_keyword_policy=expected_keyword_policy,
    )
    recall = (matched_expected / total_expected) if total_expected > 0 else None
    chunk_hit = any(item["chunk"] for item in flags)
    root_hit = any(item["root"] for item in flags)
    file_hit = any(item["file"] for item in flags)
    page_hit = any(item["page"] for item in flags)
    mrr = (1.0 / rank) if rank else 0.0
    chunk_rank = next((idx for idx, item in enumerate(flags, 1) if item["chunk"]), None)
    root_rank = next((idx for idx, item in enumerate(flags, 1) if item["root"]), None)
    chunk_mrr = (1.0 / chunk_rank) if chunk_rank else 0.0
    root_mrr = (1.0 / root_rank) if root_rank else 0.0
    file_page_hit = any(item["file"] and item["page"] for item in flags) if file_qrel_available and page_qrel_available else None
    answer_support_hit = None
    if chunk_qrel_available or root_qrel_available:
        answer_support_hit = bool(chunk_hit or (root_hit and (page_hit or any(item["anchor"] for item in flags))))

    return {
        "top_k": top_k,
        "returned_count": returned_count,
        "scorable": has_expected,
        "file_qrel_available": file_qrel_available,
        "page_qrel_available": page_qrel_available,
        "chunk_qrel_available": chunk_qrel_available,
        "root_qrel_available": root_qrel_available,
        "hit_at_5": bool(rank),
        "root_hit_at_5": root_hit if root_qrel_available else None,
        "root_mrr": root_mrr if root_qrel_available else None,
        "anchor_hit_at_5": any(item["anchor"] for item in flags),
        "keyword_hit_at_5": any(item["keyword"] for item in flags),
        "keyword_required_hit_at_5": any(item["keyword_required"] for item in flags),
        "file_hit_at_5": file_hit if file_qrel_available else None,
        "file_page_hit_at_5": file_page_hit,
        "page_hit_at_5": page_hit if page_qrel_available else None,
        "chunk_hit_at_5": chunk_hit if chunk_qrel_available else None,
        "chunk_mrr": chunk_mrr if chunk_qrel_available else None,
        "answer_support_hit_at_5_experimental": answer_support_hit,
        "legacy_chunk_hit_at_5": chunk_hit if chunk_qrel_available else None,
        "first_relevant_rank": rank,
        "mrr": mrr,
        "positive_chunk_mrr": mrr if chunk_hit else 0.0,
        "context_precision_id_at_5": precision if has_expected else None,
        "id_context_precision_at_5": id_precision if id_precision is not None else (precision if has_expected else None),
        "id_context_recall_at_5": id_recall,
        "ndcg_at_5": ndcg,
        "map_at_5": map_at_k,
        "irrelevant_context_ratio_at_5": (1.0 - precision) if has_expected else None,
        "hard_negative_file_hit_at_5": bool(hard_negative_count) if hard_negative_file_set else None,
        "hard_negative_context_ratio_at_5": hard_negative_ratio,
        "recall_at_5": recall,
        "relevant_count": relevant_count,
        "error_rate": 0.0,
    }


def compare_sample_rank(old_metrics: dict, new_metrics: dict) -> str:
    return _compare_sample_rank(old_metrics, new_metrics)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    return _load_jsonl(path, limit=limit)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    _write_jsonl(path, rows)


def summarize_results(rows: list[dict], variants: list[str]) -> dict[str, Any]:
    return _summarize_results(rows, variants, pair_definitions=PAIR_DEFINITIONS)

def render_summary_markdown(summary: dict) -> str:
    return _render_summary_markdown(summary)


def _expected_fields(record: dict) -> dict[str, Any]:
    expected_chunk_ids = (
        _as_list(record.get("expected_chunk_ids"))
        or _as_list(record.get("legacy_gold_chunk_ids"))
        or _as_list(record.get("gold_chunk_ids"))
    )
    expected_canonical_chunk_ids = (
        _as_list(record.get("expected_canonical_chunk_ids"))
        or _as_list(record.get("canonical_chunk_ids"))
    )
    expected_canonical_root_ids = (
        _as_list(record.get("expected_canonical_root_ids"))
        or _as_list(record.get("canonical_root_ids"))
    )
    return {
        "expected_chunk_ids": expected_chunk_ids,
        "expected_root_ids": _as_list(record.get("expected_root_ids")),
        "expected_canonical_chunk_ids": expected_canonical_chunk_ids,
        "expected_canonical_root_ids": expected_canonical_root_ids,
        "chunk_qrel_match_mode": str(record.get("chunk_qrel_match_mode") or "strict_id"),
        "root_qrel_match_mode": str(record.get("root_qrel_match_mode") or "strict_id"),
        "expected_anchors": _as_list(record.get("expected_anchors")),
        "expected_keywords": _as_list(record.get("expected_keywords")),
        "expected_files": _as_list(record.get("expected_files")) or _as_list(record.get("gold_files")),
        "expected_pages": _as_list(record.get("expected_pages")) or _as_list(record.get("gold_pages")),
        "expected_page_refs": _as_list(record.get("expected_page_refs")) or _as_list(record.get("gold_doc_ids")),
        "expected_all_keywords": _as_list(record.get("expected_all_keywords")),
        "expected_keyword_policy": record.get("expected_keyword_policy") if isinstance(record.get("expected_keyword_policy"), dict) else {},
        "positive_contexts": record.get("positive_contexts") if isinstance(record.get("positive_contexts"), list) else [],
        "relevance_judgments": record.get("relevance_judgments") if isinstance(record.get("relevance_judgments"), list) else [],
        "hard_negative_files": _as_list(record.get("hard_negative_files")),
    }


def _summarize_doc(doc: dict) -> dict[str, Any]:
    text = str(doc.get("text") or doc.get("retrieval_text") or doc.get("text_preview") or "")
    return {
        "chunk_id": doc.get("chunk_id"),
        "parent_chunk_id": doc.get("parent_chunk_id"),
        "root_chunk_id": doc.get("root_chunk_id"),
        "anchor_id": doc.get("anchor_id"),
        "section_title": doc.get("section_title"),
        "section_path": doc.get("section_path"),
        "filename": doc.get("filename"),
        "page_number": doc.get("page_number"),
        "page_start": doc.get("page_start"),
        "page_end": doc.get("page_end"),
        "index_profile": doc.get("index_profile"),
        "score": doc.get("score"),
        "raw_rerank_score": doc.get("raw_rerank_score"),
        "rerank_score": doc.get("rerank_score"),
        "fusion_score": doc.get("fusion_score"),
        "final_score": doc.get("final_score"),
        "text_preview": text[:240],
    }


def _stage_hit(docs: list[dict], expected: dict[str, Any], top_k: int) -> bool:
    metrics = compute_retrieval_metrics(docs or [], top_k=top_k, **expected)
    return bool(metrics.get("hit_at_5") or metrics.get("id_context_recall_at_5"))


def _first_expected_file_rank(docs: list[dict], expected_files: list[str], top_k: int | None = None) -> int | None:
    files = _normalized_filename_set(expected_files)
    if not files:
        return None
    window = docs if top_k is None else docs[:top_k]
    for idx, doc in enumerate(window or [], 1):
        if _doc_filename_norm(doc) in files:
            return idx
    return None


def _first_expected_file_page_rank(
    docs: list[dict],
    expected_files: list[str],
    expected_pages: list[str],
    top_k: int | None = None,
) -> int | None:
    files = _normalized_filename_set(expected_files)
    pages = {str(page) for page in (expected_pages or []) if str(page)}
    if not files or not pages:
        return None
    window = docs if top_k is None else docs[:top_k]
    for idx, doc in enumerate(window or [], 1):
        if _doc_filename_norm(doc) not in files:
            continue
        if _exact_page_candidates(doc) & pages:
            return idx
    return None


def _stage_metrics(meta: dict, expected: dict[str, Any], top_k: int) -> dict[str, Any]:
    before = meta.get("candidates_before_rerank") or []
    after_rerank = meta.get("candidates_after_rerank") or []
    after_structure = meta.get("candidates_after_structure_rerank") or []
    before_hit = _stage_hit(before, expected, top_k=max(top_k, len(before) or top_k))
    rerank_hit = _stage_hit(after_rerank, expected, top_k=max(top_k, len(after_rerank) or top_k))
    structure_hit = _stage_hit(after_structure, expected, top_k=top_k)
    before_file_rank = _first_expected_file_rank(before, expected.get("expected_files") or [], top_k=None)
    rerank_file_rank = _first_expected_file_rank(after_rerank, expected.get("expected_files") or [], top_k=None)
    structure_file_rank = _first_expected_file_rank(after_structure, expected.get("expected_files") or [], top_k=top_k)
    before_page_rank = _first_expected_file_page_rank(
        before,
        expected.get("expected_files") or [],
        expected.get("expected_pages") or [],
        top_k=None,
    )
    rerank_page_rank = _first_expected_file_page_rank(
        after_rerank,
        expected.get("expected_files") or [],
        expected.get("expected_pages") or [],
        top_k=None,
    )
    before_file_hit = before_file_rank is not None
    rerank_file_hit = rerank_file_rank is not None
    structure_file_hit = structure_file_rank is not None
    page_rank_delta = None
    if before_page_rank is not None and rerank_page_rank is not None:
        page_rank_delta = before_page_rank - rerank_page_rank
    return {
        "candidate_recall_before_rerank": 1.0 if before_hit else 0.0,
        "rerank_drop_rate": 1.0 if before_hit and not rerank_hit else 0.0,
        "structure_drop_rate": 1.0 if rerank_hit and not structure_hit else 0.0,
        "file_candidate_recall_before_rerank": 1.0 if before_file_hit else 0.0,
        "file_rerank_drop_rate": 1.0 if before_file_hit and not rerank_file_hit else 0.0,
        "file_structure_drop_rate": 1.0 if rerank_file_hit and not structure_file_hit else 0.0,
        "file_rank_before_rerank": before_file_rank,
        "file_rank_after_rerank": rerank_file_rank,
        "file_rank_after_structure_rerank": structure_file_rank,
        "page_rank_before_rerank": before_page_rank,
        "page_rank_after_rerank": rerank_page_rank,
        "page_rank_delta": page_rank_delta,
    }


def _metric_success(metrics: dict[str, Any]) -> bool:
    id_recall = metrics.get("id_context_recall_at_5")
    return bool(metrics.get("hit_at_5") or (isinstance(id_recall, (int, float)) and id_recall > 0))


def _graph_transition_metrics(
    initial_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    initial_hit = _metric_success(initial_metrics)
    final_hit = _metric_success(final_metrics)
    fallback_triggered = bool(
        trace.get("fallback_required")
        or trace.get("rewrite_needed")
        or trace.get("retrieval_stage") == "expanded"
    )
    fallback_executed = bool(
        trace.get("fallback_executed")
        or trace.get("rewrite_needed")
        or trace.get("retrieval_stage") == "expanded"
    )
    return {
        "initial_retrieval_hit_at_5": initial_hit,
        "final_retrieval_hit_at_5": final_hit,
        "fallback_triggered": fallback_triggered,
        "fallback_executed": fallback_executed,
        "fallback_helped": fallback_executed and not initial_hit and final_hit,
        "fallback_hurt": fallback_executed and initial_hit and not final_hit,
    }


def _ensure_backend_imports(mode: str) -> tuple[Any, Any, Any | None]:
    from backend.rag.diagnostics import classify_failure
    from backend.rag.utils import retrieve_documents

    run_rag_graph = None
    if mode == "graph":
        from backend.rag.pipeline import run_rag_graph

    return classify_failure, retrieve_documents, run_rag_graph


def _ensure_answer_eval_import() -> Any:
    from backend.evaluation.answer_eval import evaluate_answer_end_to_end

    return evaluate_answer_end_to_end

def evaluate_sample(record: dict, variant: str, top_k: int, mode: str = "retrieval") -> dict[str, Any]:
    classify_failure, retrieve_documents, run_rag_graph = _ensure_backend_imports(mode)
    query = str(record.get("question") or record.get("query") or record.get("input") or "")
    sample_id = str(record.get("sample_id") or record.get("id") or query[:60])
    expected = _expected_fields(record)
    answer_eval: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        if mode == "graph":
            if run_rag_graph is None:
                raise RuntimeError("run_rag_graph import failed")
            graph_result = run_rag_graph(query, context_files=None)
            meta = graph_result.get("rag_trace") or {}
            docs = graph_result.get("docs") or meta.get("retrieved_chunks") or []
            initial_docs = meta.get("initial_retrieved_chunks") or []
        else:
            retrieved = retrieve_documents(query, top_k=top_k, context_files=None)
            docs = retrieved.get("docs", [])
            meta = retrieved.get("meta", {})
            initial_docs = meta.get("candidates_after_structure_rerank") or docs

        if mode == "answer-eval":
            evaluate_answer_end_to_end = _ensure_answer_eval_import()
            answer_eval = evaluate_answer_end_to_end(
                question=query,
                docs=docs[:top_k],
                expected=expected,
                reference_answer=record.get("reference_answer") or record.get("expected_answer"),
            )

        latency_ms = (time.perf_counter() - started) * 1000
        metrics = compute_retrieval_metrics(docs, top_k=top_k, **expected)
        initial_metrics = compute_retrieval_metrics(initial_docs, top_k=top_k, **expected)
        if mode == "graph":
            metrics.update(_graph_transition_metrics(initial_metrics, metrics, meta))
        else:
            metrics.update(_stage_metrics(meta, expected, top_k=top_k))
            metrics.update(
                {
                    "initial_retrieval_hit_at_5": _metric_success(metrics),
                    "final_retrieval_hit_at_5": _metric_success(metrics),
                    "fallback_triggered": bool(meta.get("fallback_required")),
                    "fallback_executed": bool(meta.get("fallback_executed")),
                    "fallback_helped": False,
                    "fallback_hurt": False,
                }
            )
        if answer_eval is not None:
            metrics.update(
                {
                    "faithfulness_score": answer_eval.get("faithfulness_score"),
                    "answer_relevance_score": answer_eval.get("answer_relevance_score"),
                    "citation_coverage": answer_eval.get("citation_coverage"),
                    "answer_eval_error_rate": 1.0
                    if answer_eval.get("answer_error") or answer_eval.get("judge_error")
                    else 0.0,
                }
            )
        metrics["error_rate"] = 0.0
        rag_trace = {"retrieved_chunks": docs, **meta}
        diagnostic_expected = {
            key: expected[key]
            for key in ("expected_chunk_ids", "expected_root_ids", "expected_anchors", "expected_keywords", "expected_files", "expected_pages", "hard_negative_files")
            if key in expected
        }
        diagnostic_result = classify_failure(query=query, rag_trace=rag_trace, **diagnostic_expected)
        return {
            "sample_id": sample_id,
            "variant": variant,
            "mode": mode,
            "query": query,
            "expected": expected,
            "retrieved_chunks": [_summarize_doc(doc) for doc in docs[:top_k]],
            "initial_retrieved_chunks": [_summarize_doc(doc) for doc in initial_docs[:top_k]],
            "trace": _summarize_trace(meta),
            "metrics": metrics,
            "initial_metrics": initial_metrics,
            "answer": (answer_eval or {}).get("answer"),
            "answer_eval": answer_eval,
            "diagnostic_result": diagnostic_result,
            "latency_ms": latency_ms,
            "error_rate": 0.0,
            "fallback_required": bool(metrics.get("fallback_triggered") or meta.get("fallback_required")),
            "fallback_executed": bool(metrics.get("fallback_executed") or meta.get("fallback_executed")),
            "fallback_helped": bool(metrics.get("fallback_helped")),
            "fallback_hurt": bool(metrics.get("fallback_hurt")),
            "rewrite_strategy": meta.get("rewrite_strategy") or "none",
            "error": None,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "sample_id": sample_id,
            "variant": variant,
            "mode": mode,
            "query": query,
            "expected": expected,
            "retrieved_chunks": [],
            "initial_retrieved_chunks": [],
            "trace": {},
            "metrics": {
                "top_k": top_k,
                "returned_count": 0,
                "scorable": bool(any(expected.values())),
                "file_qrel_available": bool(expected.get("expected_files")),
                "page_qrel_available": bool(
                    expected.get("relevance_judgments")
                    or expected.get("positive_contexts")
                    or expected.get("expected_page_refs")
                    or expected.get("expected_pages")
                ),
                "chunk_qrel_available": bool(
                    expected.get("expected_chunk_ids") or expected.get("expected_canonical_chunk_ids")
                ),
                "root_qrel_available": bool(
                    expected.get("expected_root_ids") or expected.get("expected_canonical_root_ids")
                ),
                "hit_at_5": False,
                "chunk_hit_at_5": False
                if expected.get("expected_chunk_ids") or expected.get("expected_canonical_chunk_ids")
                else None,
                "root_hit_at_5": False
                if expected.get("expected_root_ids") or expected.get("expected_canonical_root_ids")
                else None,
                "chunk_mrr": 0.0
                if expected.get("expected_chunk_ids") or expected.get("expected_canonical_chunk_ids")
                else None,
                "root_mrr": 0.0
                if expected.get("expected_root_ids") or expected.get("expected_canonical_root_ids")
                else None,
                "answer_support_hit_at_5_experimental": False
                if expected.get("expected_chunk_ids")
                or expected.get("expected_canonical_chunk_ids")
                or expected.get("expected_root_ids")
                or expected.get("expected_canonical_root_ids")
                else None,
                "anchor_hit_at_5": False,
                "keyword_hit_at_5": False,
                "keyword_required_hit_at_5": False,
                "file_hit_at_5": False if expected.get("expected_files") else None,
                "file_page_hit_at_5": False
                if expected.get("expected_files")
                and (
                    expected.get("relevance_judgments")
                    or expected.get("positive_contexts")
                    or expected.get("expected_page_refs")
                    or expected.get("expected_pages")
                )
                else None,
                "page_hit_at_5": False
                if (
                    expected.get("relevance_judgments")
                    or expected.get("positive_contexts")
                    or expected.get("expected_page_refs")
                    or expected.get("expected_pages")
                )
                else None,
                "legacy_chunk_hit_at_5": False
                if expected.get("expected_chunk_ids") or expected.get("expected_canonical_chunk_ids")
                else None,
                "first_relevant_rank": None,
                "mrr": 0.0,
                "positive_chunk_mrr": 0.0,
                "context_precision_id_at_5": 0.0 if any(expected.values()) else None,
                "id_context_precision_at_5": 0.0 if any(expected.values()) else None,
                "id_context_recall_at_5": 0.0 if any(expected.values()) else None,
                "ndcg_at_5": 0.0 if any(expected.values()) else None,
                "map_at_5": 0.0 if any(expected.values()) else None,
                "irrelevant_context_ratio_at_5": 1.0 if any(expected.values()) else None,
                "hard_negative_file_hit_at_5": False if expected.get("hard_negative_files") else None,
                "hard_negative_context_ratio_at_5": 0.0 if expected.get("hard_negative_files") else None,
                "candidate_recall_before_rerank": 0.0,
                "rerank_drop_rate": 0.0,
                "structure_drop_rate": 0.0,
                "initial_retrieval_hit_at_5": False,
                "final_retrieval_hit_at_5": False,
                "fallback_triggered": False,
                "fallback_executed": False,
                "fallback_helped": False,
                "fallback_hurt": False,
                "faithfulness_score": None,
                "answer_relevance_score": None,
                "citation_coverage": None,
                "answer_eval_error_rate": 1.0 if mode == "answer-eval" else 0.0,
                "relevant_count": 0,
                "error_rate": 1.0,
            },
            "answer": None,
            "answer_eval": None,
            "diagnostic_result": {
                "category": "error",
                "failed_stage": "unknown",
                "evidence": {"error": str(exc)},
                "suggestions": ["Check retrieval runtime, Milvus connection, embedding, and rerank configuration."],
            },
            "latency_ms": latency_ms,
            "error_rate": 1.0,
            "fallback_required": False,
            "fallback_executed": False,
            "fallback_helped": False,
            "fallback_hurt": False,
            "rewrite_strategy": "none",
            "error": str(exc),
        }


def _summarize_trace(meta: dict) -> dict[str, Any]:
    keys = [
        "retrieval_mode",
        "candidate_k",
        "candidate_count_before_rerank",
        "candidate_count_after_rerank",
        "candidate_count_after_structure_rerank",
        "leaf_retrieve_level",
        "rerank_enabled",
        "rerank_applied",
        "rerank_top_n",
        "rerank_cpu_top_n_cap",
        "rerank_input_count",
        "rerank_output_count",
        "rerank_input_cap",
        "rerank_input_device_tier",
        "rerank_cache_enabled",
        "rerank_cache_hit",
        "rerank_error",
        "milvus_search_ef",
        "milvus_sparse_drop_ratio",
        "milvus_rrf_k",
        "structure_rerank_enabled",
        "structure_rerank_applied",
        "structure_rerank_root_weight",
        "same_root_cap",
        "confidence_gate_enabled",
        "fallback_required",
        "fallback_required_raw",
        "fallback_executed",
        "fallback_disabled",
        "rewrite_needed",
        "grade_route",
        "grade_score",
        "rewrite_strategy",
        "confidence_reasons",
        "top_margin",
        "top_score",
        "dominant_root_id",
        "dominant_root_share",
        "anchor_match",
        "query_anchors",
        "hybrid_error",
        "dense_error",
        "timings",
        "stage_errors",
        "context_chars",
        "retrieved_chunk_count",
        "final_context_chunk_count",
    ]
    summary = {key: meta.get(key) for key in keys if key in meta}
    for trace_key in (
        "candidates_before_rerank",
        "candidates_after_rerank",
        "candidates_after_structure_rerank",
        "initial_retrieved_chunks",
        "expanded_retrieved_chunks",
    ):
        summary[trace_key] = (meta.get(trace_key) or [])[:10]
    return summary


def evaluate_variant(dataset: Path, variant: str, output: Path, limit: int | None, top_k: int, mode: str) -> int:
    validation = validate_eval_dataset_records(load_jsonl(dataset), dataset)
    if not validation["ok"]:
        raise RuntimeError(f"dataset validation failed: {validation['errors'][:3]}")
    records = load_jsonl(dataset, limit=limit)
    rows = []
    for idx, record in enumerate(records, 1):
        row = evaluate_sample(record, variant=variant, top_k=top_k, mode=mode)
        rows.append(row)
        print(
            f"done variant={variant} sample={idx}/{len(records)} id={row['sample_id']} "
            f"hit={row['metrics'].get('hit_at_5')} err={bool(row.get('error'))}",
            flush=True,
        )
    write_jsonl(output, rows)
    return 0


def _merged_env(variant: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in VARIANT_CONFIGS[variant]["env"].items()})
    env.setdefault("PYTHONUTF8", "1")
    return env


def _run_reindex(variant: str, documents_dir: Path) -> dict[str, Any]:
    config = VARIANT_CONFIGS[variant]
    env = _merged_env(variant)
    collection = env.get("MILVUS_COLLECTION") or "embeddings_collection"
    index_profile = env.get("RAG_INDEX_PROFILE") or "legacy"
    state_path = env.get("BM25_STATE_PATH")
    text_mode = str(config["reindex_mode"])
    started_at = datetime.now().isoformat(timespec="seconds")
    print(
        f"reindex variant={variant} profile={index_profile} collection={collection} "
        f"mode={text_mode} documents_dir={documents_dir}",
        flush=True,
    )
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "reindex_knowledge_base.py"),
        "--documents-dir",
        str(documents_dir),
        "--index-profile",
        index_profile,
        "--collection",
        collection,
        "--text-mode",
        text_mode,
    ]
    if state_path:
        cmd.extend(["--state-path", state_path])
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"reindex failed for {variant} with exit code {result.returncode}")
    return {
        "variant": variant,
        "index_profile": index_profile,
        "collection": collection,
        "text_mode": text_mode,
        "state_path": state_path,
        "documents_dir": str(documents_dir),
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def _variant_requires_destructive_ack(variant: str) -> bool:
    env = _merged_env(variant)
    collection = env.get("MILVUS_COLLECTION") or "embeddings_collection"
    profile = env.get("RAG_INDEX_PROFILE") or "legacy"
    return profile == "legacy" and collection in {"embeddings_collection", DEFAULT_S3_COLLECTION}


def _run_worker(dataset: Path, report_dir: Path, variant: str, limit: int | None, top_k: int, mode: str) -> Path:
    output = report_dir / f"variant-{variant}.jsonl"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--dataset",
        str(dataset),
        "--top-k",
        str(top_k),
        "--mode",
        mode,
        "--worker-variant",
        variant,
        "--worker-output",
        str(output),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=_merged_env(variant), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"worker failed for {variant} with exit code {result.returncode}")
    return output


def _git_status_summary() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]
    except Exception as exc:
        return [f"git status failed: {exc}"]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = result.stdout.strip()
        return commit or None
    except Exception:
        return None


def _config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _variant_fingerprints(args: argparse.Namespace, variants: list[str]) -> dict[str, dict[str, Any]]:
    dataset_sha = _sha256_file(args.dataset)
    out: dict[str, dict[str, Any]] = {}
    for variant in variants:
        env = _merged_env(variant)
        payload = {
            "variant": variant,
            "variant_config": VARIANT_CONFIGS[variant],
            "dataset": str(args.dataset),
            "dataset_sha256": dataset_sha,
            "documents_dir": str(args.documents_dir),
            "top_k": args.top_k,
            "mode": args.mode,
        }
        out[variant] = {
            "index_profile": env.get("RAG_INDEX_PROFILE") or "legacy",
            "collection": env.get("MILVUS_COLLECTION") or "embeddings_collection",
            "text_mode": env.get("EVAL_RETRIEVAL_TEXT_MODE"),
            "bm25_state": env.get("BM25_STATE_PATH"),
            "embedding_provider": env.get("EMBEDDING_PROVIDER") or os.getenv("EMBEDDING_PROVIDER", "local"),
            "embedding_model": env.get("EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            "reranker_model": env.get("RERANK_MODEL") or os.getenv("RERANK_MODEL"),
            "profile_config_hash": _config_hash(payload),
        }
    return out


def _build_fingerprint(args: argparse.Namespace, variants: list[str], reindex_events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "git_status": _git_status_summary(),
        "corpus_path": str(args.documents_dir),
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256_file(args.dataset),
        "index_build_timestamp": reindex_events[-1]["completed_at"] if reindex_events else None,
        "reindex_events": reindex_events,
        "variants": _variant_fingerprints(args, variants),
    }


def _env_snapshot(keys: list[str]) -> dict[str, str | None]:
    return {key: os.getenv(key) for key in keys}


def _cuda_snapshot() -> dict[str, Any]:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        return {
            "available": available,
            "device_name": torch.cuda.get_device_name(0) if available else "NO_CUDA",
            "torch_version": getattr(torch, "__version__", None),
        }
    except Exception as exc:
        return {"available": False, "device_name": "UNKNOWN", "error": str(exc)}


def _write_config(
    path: Path,
    args: argparse.Namespace,
    variants: list[str],
    destructive_reindex_run: bool,
    reindex_events: list[dict[str, Any]],
) -> None:
    dataset_records = load_jsonl(args.dataset)
    dataset_validation = validate_eval_dataset_records(dataset_records, args.dataset)
    env_keys = [
        "EVAL_RETRIEVAL_TEXT_MODE",
        "MILVUS_COLLECTION",
        "STRUCTURE_RERANK_ENABLED",
        "CONFIDENCE_GATE_ENABLED",
        "LOW_CONF_TOP_MARGIN",
        "LOW_CONF_ROOT_SHARE",
        "LOW_CONF_TOP_SCORE",
        "ENABLE_ANCHOR_GATE",
        "RERANK_PROVIDER",
        "RERANK_MODEL",
        "RERANK_DEVICE",
        "RERANK_CPU_TOP_N_CAP",
        "RERANK_INPUT_K_CPU",
        "RERANK_INPUT_K_GPU",
        "RAG_CANDIDATE_K",
        "RERANK_TOP_N",
        "MILVUS_SEARCH_EF",
        "MILVUS_SPARSE_DROP_RATIO",
        "MILVUS_RRF_K",
        "QUERY_PLAN_ENABLED",
        "DOC_SCOPE_MATCH_FILTER",
        "DOC_SCOPE_MATCH_BOOST",
        "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT",
        "HEADING_LEXICAL_ENABLED",
        "HEADING_LEXICAL_WEIGHT",
        "RERANK_PAIR_ENRICHMENT_ENABLED",
        "RAG_INDEX_PROFILE",
        "RERANK_SCORE_FUSION_ENABLED",
        "RERANK_FUSION_RERANK_WEIGHT",
        "RERANK_FUSION_RRF_WEIGHT",
        "RERANK_FUSION_SCOPE_WEIGHT",
        "RERANK_FUSION_METADATA_WEIGHT",
        "RERANK_BLEND_ALPHA",
        "BM25_STATE_PATH",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_DEVICE",
        "ANSWER_EVAL_GENERATION_MODEL",
        "ANSWER_EVAL_JUDGE_MODEL",
        "FAST_MODEL",
    ]
    config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_schema_version": EVAL_SCHEMA_VERSION,
        "dataset_profile": args.dataset_profile,
        "dataset": str(args.dataset),
        "documents_dir": str(args.documents_dir),
        "dataset_sha256": _sha256_file(args.dataset),
        "dataset_row_count": len(dataset_records),
        "dataset_schema_versions": _dataset_schema_versions(dataset_records),
        "dataset_validation": dataset_validation,
        "mode": args.mode,
        "limit": args.limit,
        "top_k": args.top_k,
        "variants": {variant: VARIANT_CONFIGS[variant] for variant in variants},
        "destructive_reindex_run": destructive_reindex_run,
        "reindex_events": reindex_events,
        "build_fingerprint": _build_fingerprint(args, variants, reindex_events),
        "skip_reindex": args.skip_reindex,
        "skip_coverage_check": args.skip_coverage_check,
        "coverage_threshold": args.coverage_threshold,
        "chunk_qrels": str(args.chunk_qrels) if args.chunk_qrels else None,
        "chunk_qrel_match_mode": args.chunk_qrel_match_mode,
        "root_qrel_match_mode": args.root_qrel_match_mode,
        "qrel_conflict_policy": args.qrel_conflict_policy,
        "require_chunk_qrel_coverage": args.require_chunk_qrel_coverage,
        "require_root_qrel_coverage": args.require_root_qrel_coverage,
        "git_status": _git_status_summary(),
        "cuda": _cuda_snapshot(),
        "env_keys": env_keys,
        "env_snapshot": _env_snapshot(env_keys),
    }
    config["cpu_only_run"] = not bool(config["cuda"].get("available"))
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _miss_analysis_rows(rows: list[dict]) -> list[dict]:
    misses: list[dict] = []
    for row in rows:
        metrics = row.get("metrics") or {}
        if metrics.get("hit_at_5") or metrics.get("id_context_recall_at_5"):
            continue
        diagnostic = row.get("diagnostic_result") or {}
        trace = row.get("trace") or {}
        misses.append(
            {
                "sample_id": row.get("sample_id"),
                "variant": row.get("variant"),
                "query": row.get("query"),
                "category": diagnostic.get("category"),
                "failed_stage": diagnostic.get("failed_stage"),
                "expected": row.get("expected"),
                "metrics": {
                    "file_hit_at_5": metrics.get("file_hit_at_5"),
                    "file_page_hit_at_5": metrics.get("file_page_hit_at_5"),
                    "page_hit_at_5": metrics.get("page_hit_at_5"),
                    "keyword_required_hit_at_5": metrics.get("keyword_required_hit_at_5"),
                    "context_precision_id_at_5": metrics.get("context_precision_id_at_5"),
                    "id_context_recall_at_5": metrics.get("id_context_recall_at_5"),
                    "candidate_recall_before_rerank": metrics.get("candidate_recall_before_rerank"),
                    "file_candidate_recall_before_rerank": metrics.get("file_candidate_recall_before_rerank"),
                    "rerank_drop_rate": metrics.get("rerank_drop_rate"),
                    "file_rerank_drop_rate": metrics.get("file_rerank_drop_rate"),
                    "file_rank_before_rerank": metrics.get("file_rank_before_rerank"),
                    "page_rank_before_rerank": metrics.get("page_rank_before_rerank"),
                    "page_rank_after_rerank": metrics.get("page_rank_after_rerank"),
                    "page_rank_delta": metrics.get("page_rank_delta"),
                    "structure_drop_rate": metrics.get("structure_drop_rate"),
                    "initial_retrieval_hit_at_5": metrics.get("initial_retrieval_hit_at_5"),
                    "final_retrieval_hit_at_5": metrics.get("final_retrieval_hit_at_5"),
                    "fallback_triggered": metrics.get("fallback_triggered"),
                    "fallback_executed": metrics.get("fallback_executed"),
                    "fallback_helped": metrics.get("fallback_helped"),
                    "fallback_hurt": metrics.get("fallback_hurt"),
                },
                "retrieved_chunks": row.get("retrieved_chunks"),
                "initial_retrieved_chunks": row.get("initial_retrieved_chunks"),
                "stage_candidates": {
                    "before_rerank": trace.get("candidates_before_rerank", []),
                    "after_rerank": trace.get("candidates_after_rerank", []),
                    "after_structure_rerank": trace.get("candidates_after_structure_rerank", []),
                },
                "rewrite_strategy": row.get("rewrite_strategy"),
                "suggestions": diagnostic.get("suggestions") or [],
            }
        )
    return misses


def _saved_results_variants(rows: list[dict], raw_variants: str) -> list[str]:
    row_variants = sorted({str(row.get("variant")) for row in rows if row.get("variant")})
    if not row_variants:
        return []
    if raw_variants == DEFAULT_VARIANTS:
        return row_variants
    requested = parse_variants(raw_variants)
    return [variant for variant in requested if variant in row_variants]


def run_saved_results_summary(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.summarize_results_jsonl, limit=args.limit)
    variants = _saved_results_variants(rows, args.variants)
    if not variants:
        raise RuntimeError(f"no variants found in saved results: {args.summarize_results_jsonl}")

    report_dir = args.output_root / args.run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(rows, variants=variants)
    summary["build_fingerprint"] = {
        "git_commit": _git_commit(),
        "git_status": _git_status_summary(),
        "source_results_jsonl": str(args.summarize_results_jsonl),
        "source_results_sha256": _sha256_file(args.summarize_results_jsonl),
        "variants": variants,
        "mode": "saved_results_summary",
    }
    summary["qrel_contract"] = {
        "mode": "saved_results_summary",
        "source_results_jsonl": str(args.summarize_results_jsonl),
    }

    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (report_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")

    if args.regression_baseline_summary:
        baseline = json.loads(args.regression_baseline_summary.read_text(encoding="utf-8"))
        diffs = compare_core_summary(baseline, summary)
        regression_report = {
            "baseline_summary": str(args.regression_baseline_summary),
            "generated_summary": str(report_dir / "summary.json"),
            "diff_count": len(diffs),
            "diffs": diffs,
        }
        (report_dir / "summary_regression.json").write_text(
            json.dumps(regression_report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if diffs and args.regression_fail_on_diff:
            raise RuntimeError(
                f"saved-row summary regression found {len(diffs)} core metric differences; "
                f"see {report_dir / 'summary_regression.json'}"
            )

    print(f"wrote saved-results summary {report_dir}", flush=True)
    return 0


def run_matrix(args: argparse.Namespace) -> int:
    variants = parse_variants(args.variants)
    report_dir = args.output_root / args.run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_records = load_jsonl(args.dataset)
    original_dataset = args.dataset
    qrel_merge_report: dict[str, Any] | None = None
    if args.chunk_qrels:
        qrel_rows = load_chunk_qrels(args.chunk_qrels)
        dataset_records, qrel_merge_report = merge_chunk_qrels(
            dataset_records,
            qrel_rows,
            conflict_policy=args.qrel_conflict_policy,
            chunk_match_mode=args.chunk_qrel_match_mode,
            root_match_mode=args.root_qrel_match_mode,
            qrel_source=str(args.chunk_qrels),
        )
        merged_dataset = report_dir / "dataset_with_chunk_qrels.jsonl"
        write_jsonl(merged_dataset, dataset_records)
        args.dataset = merged_dataset.resolve()
    validation = validate_eval_dataset_records(dataset_records, args.dataset)
    if not validation["ok"]:
        (report_dir / "dataset_validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"dataset validation failed for {args.dataset}; "
            f"first errors: {validation['errors'][:3]}"
        )
    corpus_coverage = _corpus_coverage_report(dataset_records, args.documents_dir)
    if corpus_coverage["coverage"] < args.coverage_threshold:
        (report_dir / "corpus_coverage.json").write_text(
            json.dumps(corpus_coverage, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"canonical corpus coverage {corpus_coverage['coverage']:.3f} below "
            f"threshold {args.coverage_threshold:.3f}; missing examples: "
            f"{corpus_coverage['missing_files'][:5]}"
        )

    qrel_coverage = _qrel_coverage_report(dataset_records)
    if qrel_coverage["chunk_qrel_coverage"] < args.require_chunk_qrel_coverage:
        raise RuntimeError(
            f"chunk qrel coverage {qrel_coverage['chunk_qrel_coverage']:.3f} below "
            f"threshold {args.require_chunk_qrel_coverage:.3f}"
        )
    if qrel_coverage["root_qrel_coverage"] < args.require_root_qrel_coverage:
        raise RuntimeError(
            f"root qrel coverage {qrel_coverage['root_qrel_coverage']:.3f} below "
            f"threshold {args.require_root_qrel_coverage:.3f}"
        )
    destructive_reindex_run = False
    reindex_events: list[dict[str, Any]] = []
    variant_outputs: list[Path] = []
    last_reindex_mode = None
    coverage_reports: list[dict[str, Any]] = [corpus_coverage]
    if qrel_merge_report:
        coverage_reports.append(qrel_merge_report)
    coverage_reports.append(qrel_coverage)
    for variant in variants:
        config = VARIANT_CONFIGS[variant]
        if config["requires_reindex"] and not args.skip_reindex:
            needs_destructive_ack = _variant_requires_destructive_ack(variant)
            if needs_destructive_ack and not args.allow_destructive_reindex:
                raise RuntimeError(
                    f"{variant} requires destructive reindex. Pass --allow-destructive-reindex or --skip-reindex."
                )
            reindex_events.append(_run_reindex(variant, documents_dir=args.documents_dir))
            destructive_reindex_run = destructive_reindex_run or needs_destructive_ack
            last_reindex_mode = config["reindex_mode"]
        elif config["requires_reindex"] and args.skip_reindex:
            print(f"skip reindex variant={variant}", flush=True)
        elif last_reindex_mode and config["reindex_mode"] != last_reindex_mode:
            print(
                f"warning variant={variant} expects mode={config['reindex_mode']} but last reindex={last_reindex_mode}",
                flush=True,
            )

        if not args.skip_coverage_check:
            collection_coverage = _collection_coverage_report(dataset_records, variant)
            coverage_reports.append(collection_coverage)
            if collection_coverage["coverage"] < args.coverage_threshold:
                (report_dir / "coverage_preflight.json").write_text(
                    json.dumps(coverage_reports, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                raise RuntimeError(
                    f"{variant} collection coverage {collection_coverage['coverage']:.3f} below "
                    f"threshold {args.coverage_threshold:.3f}; collection={collection_coverage['collection']} "
                    f"missing examples: {collection_coverage['missing_files'][:5]}"
                )
            metadata_coverage = collection_coverage.get("metadata_coverage") or {}
            if metadata_coverage.get("hard_gate_failures"):
                (report_dir / "coverage_preflight.json").write_text(
                    json.dumps(coverage_reports, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                raise RuntimeError(
                    f"{variant} metadata preflight failed for collection={collection_coverage['collection']}: "
                    f"{metadata_coverage['hard_gate_failures']}"
                )

        variant_outputs.append(_run_worker(args.dataset, report_dir, variant, args.limit, args.top_k, args.mode))

    rows: list[dict] = []
    for output in variant_outputs:
        rows.extend(load_jsonl(output))

    write_jsonl(report_dir / "results.jsonl", rows)
    write_jsonl(report_dir / "miss_analysis.jsonl", _miss_analysis_rows(rows))
    summary = summarize_results(rows, variants=variants)
    summary["coverage_preflight"] = coverage_reports
    summary["build_fingerprint"] = _build_fingerprint(args, variants, reindex_events)
    summary["qrel_contract"] = {
        "original_dataset": str(original_dataset),
        "effective_dataset": str(args.dataset),
        "chunk_qrels": str(args.chunk_qrels) if args.chunk_qrels else None,
        "chunk_qrel_match_mode": args.chunk_qrel_match_mode,
        "root_qrel_match_mode": args.root_qrel_match_mode,
        "qrel_conflict_policy": args.qrel_conflict_policy,
        "require_chunk_qrel_coverage": args.require_chunk_qrel_coverage,
        "require_root_qrel_coverage": args.require_root_qrel_coverage,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (report_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    (report_dir / "coverage_preflight.json").write_text(
        json.dumps(coverage_reports, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _write_config(report_dir / "config.json", args, variants, destructive_reindex_run, reindex_events)
    print(f"wrote report {report_dir}", flush=True)
    return 0


def parse_variants(raw: str) -> list[str]:
    canonical_by_upper = {variant.upper(): variant for variant in VARIANT_CONFIGS}
    variants = [canonical_by_upper.get(item.strip().upper(), item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in VARIANT_CONFIGS]
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")
    return variants


def validate_variant_order(variants: list[str], skip_reindex: bool) -> None:
    if skip_reindex:
        return

    current_index_mode = None
    for variant in variants:
        config = VARIANT_CONFIGS[variant]
        expected_mode = config["reindex_mode"]
        if config["requires_reindex"]:
            current_index_mode = expected_mode
            continue
        if current_index_mode != expected_mode:
            raise RuntimeError(
                f"{variant} expects {expected_mode} index. Include a preceding reindex variant such as A1 "
                "or pass --skip-reindex when intentionally reusing an existing index."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SuperHermes RAG matrix retrieval evaluation.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--dataset-profile", choices=["frozen", "gold", "smoke", "natural", "custom"], default="frozen")
    parser.add_argument("--documents-dir", type=Path, default=DEFAULT_CANONICAL_CORPUS)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / ".jbeval" / "reports")
    parser.add_argument("--run-id", default=f"rag-matrix-{datetime.now().strftime('%Y%m%d-%H%M')}")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["retrieval", "graph", "answer-eval"], default="retrieval")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--skip-reindex", action="store_true")
    parser.add_argument("--skip-coverage-check", action="store_true")
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--allow-destructive-reindex", action="store_true")
    parser.add_argument("--chunk-qrels", type=Path, default=None)
    parser.add_argument("--chunk-qrel-match-mode", choices=sorted(VALID_QREL_MATCH_MODES), default="strict_id")
    parser.add_argument("--root-qrel-match-mode", choices=sorted(VALID_QREL_MATCH_MODES), default="strict_id")
    parser.add_argument("--qrel-conflict-policy", choices=sorted(VALID_QREL_CONFLICT_POLICIES), default="external")
    parser.add_argument("--require-chunk-qrel-coverage", type=float, default=0.0)
    parser.add_argument("--require-root-qrel-coverage", type=float, default=0.0)
    parser.add_argument("--summarize-results-jsonl", type=Path, default=None)
    parser.add_argument("--regression-baseline-summary", type=Path, default=None)
    parser.add_argument("--regression-fail-on-diff", action="store_true")
    parser.add_argument("--worker-variant", choices=sorted(VARIANT_CONFIGS), default=None)
    parser.add_argument("--worker-output", type=Path, default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = _resolve_dataset_profile(args.dataset_profile)
    elif args.dataset_profile != "custom":
        args.dataset_profile = "custom"
    args.dataset = args.dataset.resolve()
    args.documents_dir = args.documents_dir.resolve()
    args.output_root = args.output_root.resolve()
    if args.summarize_results_jsonl:
        args.summarize_results_jsonl = args.summarize_results_jsonl.resolve()
    if args.regression_baseline_summary:
        args.regression_baseline_summary = args.regression_baseline_summary.resolve()
    if args.dataset_profile == "smoke" and args.limit is None:
        args.limit = 10

    if args.summarize_results_jsonl:
        return run_saved_results_summary(args)

    if args.worker_variant:
        if args.worker_output is None:
            parser.error("--worker-output is required with --worker-variant")
        if args.chunk_qrels:
            worker_records, _ = merge_chunk_qrels(
                load_jsonl(args.dataset),
                load_chunk_qrels(args.chunk_qrels),
                conflict_policy=args.qrel_conflict_policy,
                chunk_match_mode=args.chunk_qrel_match_mode,
                root_match_mode=args.root_qrel_match_mode,
                qrel_source=str(args.chunk_qrels),
            )
            worker_dataset = args.worker_output.resolve().with_suffix(".dataset.jsonl")
            write_jsonl(worker_dataset, worker_records)
            args.dataset = worker_dataset
        return evaluate_variant(
            dataset=args.dataset,
            variant=args.worker_variant,
            output=args.worker_output.resolve(),
            limit=args.limit,
            top_k=args.top_k,
            mode=args.mode,
        )

    validate_variant_order(parse_variants(args.variants), skip_reindex=args.skip_reindex)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
