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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
DATASET_DIR = PROJECT_ROOT / ".jbeval" / "datasets"
DEFAULT_FROZEN_DATASET = DATASET_DIR / "rag_doc_frozen_eval_v1.jsonl"
DEFAULT_GOLD_DATASET = DATASET_DIR / "rag_doc_gold.jsonl"
EVAL_SCHEMA_VERSION = "rag-eval-matrix-v2"


VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "A0": {
        "description": "raw text baseline",
        "reindex_mode": "raw",
        "requires_reindex": True,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "raw",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "A1": {
        "description": "title-context retrieval text",
        "reindex_mode": "title_context",
        "requires_reindex": True,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "B1": {
        "description": "title-context retrieval text with structure rerank",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "G0": {
        "description": "structure rerank without confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "G1": {
        "description": "recommended confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.05",
            "LOW_CONF_ROOT_SHARE": "0.45",
            "LOW_CONF_TOP_SCORE": "0.20",
            "ENABLE_ANCHOR_GATE": "true",
        },
    },
    "G2": {
        "description": "looser confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.03",
            "LOW_CONF_ROOT_SHARE": "0.35",
            "LOW_CONF_TOP_SCORE": "0.15",
            "ENABLE_ANCHOR_GATE": "true",
        },
    },
    "G3": {
        "description": "stricter confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.08",
            "LOW_CONF_ROOT_SHARE": "0.55",
            "LOW_CONF_TOP_SCORE": "0.25",
            "ENABLE_ANCHOR_GATE": "true",
        },
    },
    "B0_legacy": {
        "description": "current production configuration without evaluation env overrides",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {},
    },
    "B0": {
        "description": "current title-context baseline with configured reranker at final top_k depth",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "0",
        },
    },
    "R1": {
        "description": "configured reranker with deeper rerank top_n=20",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "20",
        },
    },
    "R2": {
        "description": "larger candidate set, deeper rerank, higher Milvus ef",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_CANDIDATE_K": "80",
            "RERANK_TOP_N": "30",
            "MILVUS_SEARCH_EF": "128",
            "MILVUS_RRF_K": "60",
        },
    },
    "P1": {
        "description": "B0 retrieval depth with lower sparse drop ratio",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "0",
            "MILVUS_SPARSE_DROP_RATIO": "0.1",
            "MILVUS_RRF_K": "60",
        },
    },
    "P2": {
        "description": "B0 retrieval depth with larger RRF k",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "0",
            "MILVUS_SPARSE_DROP_RATIO": "0.2",
            "MILVUS_RRF_K": "100",
        },
    },
    "P3": {
        "description": "B0 retrieval depth with lower sparse drop ratio and larger RRF k",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "0",
            "MILVUS_SPARSE_DROP_RATIO": "0.1",
            "MILVUS_RRF_K": "100",
        },
    },
    "F1": {
        "description": "B0 retrieval baseline with confidence gate and fallback enabled",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.05",
            "LOW_CONF_ROOT_SHARE": "0.45",
            "LOW_CONF_TOP_SCORE": "0.20",
            "ENABLE_ANCHOR_GATE": "true",
            "RERANK_TOP_N": "0",
        },
    },
    "S0": {
        "description": "best retrieval candidate with structure rerank comparison",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "20",
        },
    },
    "S1_linear": {
        "description": "linear path: gate off + fallback off",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
        },
    },
    "S1": {
        "description": "compatibility alias for S1_linear",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
        },
    },
    "S2": {
        "description": "QueryPlan + doc scope 80/20 parallel",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "DOC_SCOPE_MATCH_FILTER": "0.85",
            "DOC_SCOPE_MATCH_BOOST": "0.60",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.2",
        },
    },
    "S2H": {
        "description": "S2 + heading lexical scoring",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "DOC_SCOPE_MATCH_FILTER": "0.85",
            "DOC_SCOPE_MATCH_BOOST": "0.60",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.2",
            "HEADING_LEXICAL_ENABLED": "true",
            "HEADING_LEXICAL_WEIGHT": "0.20",
        },
    },
    "S2HR": {
        "description": "S2H + metadata-aware rerank pair enrichment",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "DOC_SCOPE_MATCH_FILTER": "0.85",
            "DOC_SCOPE_MATCH_BOOST": "0.60",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.2",
            "HEADING_LEXICAL_ENABLED": "true",
            "HEADING_LEXICAL_WEIGHT": "0.20",
            "RERANK_PAIR_ENRICHMENT_ENABLED": "true",
        },
    },
    "S3": {
        "description": "full: reindex title_context_filename + all enrichments",
        "reindex_mode": "title_context_filename",
        "requires_reindex": True,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context_filename",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "DOC_SCOPE_MATCH_FILTER": "0.85",
            "DOC_SCOPE_MATCH_BOOST": "0.60",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.2",
            "HEADING_LEXICAL_ENABLED": "true",
            "HEADING_LEXICAL_WEIGHT": "0.20",
            "RERANK_PAIR_ENRICHMENT_ENABLED": "true",
        },
    },
}

PAIR_DEFINITIONS = (
    ("A1_vs_A0", "A0", "A1"),
    ("B1_vs_A1", "A1", "B1"),
    ("G1_vs_G0", "G0", "G1"),
    ("G1_vs_G2", "G2", "G1"),
    ("G1_vs_G3", "G3", "G1"),
    ("R1_vs_B0", "B0", "R1"),
    ("R2_vs_R1", "R1", "R2"),
    ("P1_vs_B0", "B0", "P1"),
    ("P2_vs_B0", "B0", "P2"),
    ("P3_vs_B0", "B0", "P3"),
    ("F1_vs_B0", "B0", "F1"),
    ("S0_vs_R1", "R1", "S0"),
    ("S1_linear_vs_B0_legacy", "B0_legacy", "S1_linear"),
    ("S2_vs_S1_linear", "S1_linear", "S2"),
    ("S2H_vs_S2", "S2", "S2H"),
    ("S2HR_vs_S2H", "S2H", "S2HR"),
    ("S3_vs_S2HR", "S2HR", "S3"),
)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


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
    errors: list[dict[str, Any]] = []
    required_any = {
        "query": ("query", "question", "input"),
        "reference_answer": ("reference_answer", "expected_answer"),
        "expected_files": ("expected_files", "gold_files"),
        "expected_pages": ("expected_pages", "expected_page_refs", "gold_pages", "gold_doc_ids"),
    }
    required_exact = ("positive_contexts", "relevance_judgments", "hard_negative_files", "expected_keyword_policy")

    for idx, record in enumerate(records, 1):
        row_id = record.get("sample_id") or record.get("id") or idx
        for label, candidates in required_any.items():
            if not any(record.get(name) for name in candidates):
                errors.append({"row": idx, "id": row_id, "error": f"missing_{label}"})
        for field in required_exact:
            if not record.get(field):
                errors.append({"row": idx, "id": row_id, "error": f"missing_{field}"})

        policy = record.get("expected_keyword_policy")
        if policy and not isinstance(policy, dict):
            errors.append({"row": idx, "id": row_id, "error": "expected_keyword_policy_not_object"})
        elif isinstance(policy, dict):
            try:
                min_match = int(policy.get("min_match", 0))
                total = int(policy.get("total", 0))
            except (TypeError, ValueError):
                errors.append({"row": idx, "id": row_id, "error": "expected_keyword_policy_invalid_numbers"})
            else:
                if min_match < 1 or (total and min_match > total):
                    errors.append({"row": idx, "id": row_id, "error": "expected_keyword_policy_invalid_range"})

    return {
        "ok": not errors,
        "dataset": str(dataset),
        "row_count": len(records),
        "schema_versions": _dataset_schema_versions(records),
        "errors": errors[:50],
        "error_count": len(errors),
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
    raw_page = doc.get("page_number")
    pages: set[str] = set()
    try:
        page = int(raw_page)
    except (TypeError, ValueError):
        return pages
    pages.add(str(page))
    pages.add(str(page + 1))
    return pages


def _doc_page_refs(doc: dict) -> set[str]:
    filename = str(doc.get("filename") or "")
    return {f"{filename}::p{page}" for page in _doc_page_candidates(doc)}


def _doc_qrel_ids(doc: dict) -> set[str]:
    ids = {str(doc.get("chunk_id") or ""), str(doc.get("root_chunk_id") or "")}
    ids |= _doc_page_refs(doc)
    return {item for item in ids if item}


def _qrel_ids_from_contexts(contexts: Any) -> set[str]:
    ids: set[str] = set()
    for item in contexts or []:
        if not isinstance(item, dict):
            continue
        for key in ("doc_id", "chunk_id", "root_chunk_id"):
            value = item.get(key)
            if value:
                ids.add(str(value))
        file_name = item.get("file_name") or item.get("filename")
        page_number = item.get("page_number")
        if file_name and page_number is not None:
            ids.add(f"{file_name}::p{page_number}")
    return ids


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
        mapped[str(doc_id)] = max(mapped.get(str(doc_id), 0.0), relevance)
    return mapped


def _doc_relevance(doc: dict, qrels: dict[str, float], seen_qrels: set[str] | None = None) -> tuple[float, str | None]:
    best_id = None
    best_score = 0.0
    for qrel_id in _doc_qrel_ids(doc):
        if seen_qrels is not None and qrel_id in seen_qrels:
            continue
        score = qrels.get(qrel_id, 0.0)
        if score > best_score:
            best_id = qrel_id
            best_score = score
    return best_score, best_id


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
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    expected_files: list[str] | None = None,
    expected_pages: list[str] | None = None,
    expected_page_refs: list[str] | None = None,
    expected_all_keywords: list[str] | None = None,
    expected_keyword_policy: dict[str, Any] | None = None,
) -> dict[str, bool]:
    chunk_ids = set(_as_list(expected_chunk_ids))
    root_ids = set(_as_list(expected_root_ids))
    anchors = _as_list(expected_anchors)
    keywords = _as_list(expected_keywords)
    all_keywords = _as_list(expected_all_keywords) or keywords
    files = set(_as_list(expected_files))
    pages = set(_as_list(expected_pages))
    page_refs = set(_as_list(expected_page_refs))
    text = _doc_text(doc)

    chunk_hit = bool(str(doc.get("chunk_id") or "") in chunk_ids)
    root_hit = bool(str(doc.get("root_chunk_id") or "") in root_ids)
    anchor_hit = any(_anchor_match(anchor, text) for anchor in anchors)
    keyword_hit = any(keyword and keyword in text for keyword in keywords)
    keyword_matches = sum(1 for keyword in all_keywords if keyword and keyword in text)
    min_keyword_match = int((expected_keyword_policy or {}).get("min_match") or (1 if all_keywords else 0))
    keyword_required_hit = bool(all_keywords) and keyword_matches >= min_keyword_match
    file_hit = bool(str(doc.get("filename") or "") in files)
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
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    expected_files: list[str] | None = None,
    expected_pages: list[str] | None = None,
    expected_page_refs: list[str] | None = None,
    expected_all_keywords: list[str] | None = None,
    expected_keyword_policy: dict[str, Any] | None = None,
    top_k: int = 5,
) -> int | None:
    prefer_chunk = bool(_as_list(expected_chunk_ids))
    for idx, doc in enumerate((docs or [])[:top_k], 1):
        flags = _doc_match_flags(
            doc,
            expected_chunk_ids=expected_chunk_ids,
            expected_root_ids=expected_root_ids,
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
    expected_anchors: list[str] | None = None,
    expected_keywords: list[str] | None = None,
    expected_files: list[str] | None = None,
    expected_pages: list[str] | None = None,
    expected_page_refs: list[str] | None = None,
    expected_all_keywords: list[str] | None = None,
    expected_keyword_policy: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Return (matched_count, total_count) for distinct expected items found in top docs."""
    chunk_ids = set(_as_list(expected_chunk_ids))
    root_ids = set(_as_list(expected_root_ids))
    anchors = _as_list(expected_anchors)
    keywords = _as_list(expected_keywords)
    files = set(_as_list(expected_files))
    pages = set(_as_list(expected_pages))
    page_refs = set(_as_list(expected_page_refs))
    all_keywords = _as_list(expected_all_keywords) or keywords

    total = len(chunk_ids) + len(root_ids) + len(anchors) + len(keywords) + len(files) + len(pages) + len(page_refs)
    if total == 0:
        return 0, 0

    found = 0
    doc_chunk_ids = {str(doc.get("chunk_id") or "") for doc in top_docs}
    doc_root_ids = {str(doc.get("root_chunk_id") or "") for doc in top_docs}
    doc_files = {str(doc.get("filename") or "") for doc in top_docs}
    doc_pages = set().union(*(_doc_page_candidates(doc) for doc in top_docs)) if top_docs else set()
    doc_page_refs = set().union(*(_doc_page_refs(doc) for doc in top_docs)) if top_docs else set()
    found += len(chunk_ids & doc_chunk_ids)
    found += len(root_ids & doc_root_ids)
    found += len(files & doc_files)
    found += len(pages & doc_pages)
    found += len(page_refs & doc_page_refs)

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
        positive_qrel_ids = set(_as_list(expected_page_refs))
    hard_negative_file_set = set(_as_list(hard_negative_files))
    has_expected = bool(
        _as_list(expected_chunk_ids)
        or _as_list(expected_root_ids)
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
    prefer_chunk = bool(_as_list(expected_chunk_ids))
    relevant_count = sum(1 for item in flags if item["chunk"]) if prefer_chunk else sum(1 for item in flags if item["any"])
    rank = first_relevant_rank(
        top_docs,
        expected_chunk_ids=expected_chunk_ids,
        expected_root_ids=expected_root_ids,
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

    positive_qrel_count = len(judgment_qrels)
    id_precision = (len(id_matches) / returned_count) if returned_count and positive_qrel_count else None
    id_recall = (len(id_matches) / positive_qrel_count) if positive_qrel_count else None
    ideal_relevances = sorted(judgment_qrels.values(), reverse=True)[:top_k]
    ideal_dcg = _discounted_cumulative_gain(ideal_relevances)
    ndcg = (_discounted_cumulative_gain(id_relevance_by_rank) / ideal_dcg) if ideal_dcg else None
    map_at_k = (average_precision_total / positive_qrel_count) if positive_qrel_count else None
    hard_negative_count = sum(1 for doc in top_docs if str(doc.get("filename") or "") in hard_negative_file_set)
    hard_negative_ratio = (hard_negative_count / returned_count) if returned_count and hard_negative_file_set else None

    matched_expected, total_expected = _count_expected_matches(
        top_docs,
        expected_chunk_ids=expected_chunk_ids,
        expected_root_ids=expected_root_ids,
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
    mrr = (1.0 / rank) if rank else 0.0
    file_page_hit = any(item["file"] and item["page"] for item in flags)

    return {
        "top_k": top_k,
        "returned_count": returned_count,
        "scorable": has_expected,
        "hit_at_5": bool(rank),
        "root_hit_at_5": any(item["root"] for item in flags),
        "anchor_hit_at_5": any(item["anchor"] for item in flags),
        "keyword_hit_at_5": any(item["keyword"] for item in flags),
        "keyword_required_hit_at_5": any(item["keyword_required"] for item in flags),
        "file_hit_at_5": any(item["file"] for item in flags),
        "file_page_hit_at_5": file_page_hit,
        "page_hit_at_5": any(item["page"] for item in flags),
        "chunk_hit_at_5": chunk_hit,
        "legacy_chunk_hit_at_5": chunk_hit,
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
    old_hit = bool(old_metrics.get("hit_at_5"))
    new_hit = bool(new_metrics.get("hit_at_5"))
    old_rank = old_metrics.get("first_relevant_rank")
    new_rank = new_metrics.get("first_relevant_rank")

    if not old_hit and new_hit:
        return "win"
    if old_hit and not new_hit:
        return "loss"
    if old_hit and new_hit:
        if new_rank < old_rank:
            return "win"
        if new_rank > old_rank:
            return "loss"
    return "tie"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _average(rows: list[dict], metric: str) -> float | None:
    values = [
        row.get("metrics", {}).get(metric)
        for row in rows
        if isinstance(row.get("metrics", {}).get(metric), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _average_field(rows: list[dict], field: str) -> float | None:
    values = [row.get(field) for row in rows if isinstance(row.get(field), (int, float, bool))]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _percentile_field(rows: list[dict], field: str, percentile: float) -> float | None:
    values = sorted(float(row.get(field)) for row in rows if isinstance(row.get(field), (int, float, bool)))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _build_pairwise(rows: list[dict], old_variant: str, new_variant: str) -> dict[str, int]:
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_sample[str(row.get("sample_id"))][str(row.get("variant"))] = row

    counts = Counter({"wins": 0, "losses": 0, "ties": 0, "missing": 0})
    for variants in by_sample.values():
        old_row = variants.get(old_variant)
        new_row = variants.get(new_variant)
        if not old_row or not new_row:
            counts["missing"] += 1
            continue
        result = compare_sample_rank(old_row.get("metrics", {}), new_row.get("metrics", {}))
        counts[{"win": "wins", "loss": "losses", "tie": "ties"}[result]] += 1
    return dict(counts)


_QRELS_NA = "n/a (qrels missing)"

# Primary dashboard metrics - Chunk@5 and Root@5 are not scorable without qrels
_PRIMARY_METRICS = [
    "file_hit_at_5",
    "file_page_hit_at_5",
    "candidate_recall_before_rerank",
    "hard_negative_context_ratio_at_5",
    "anchor_hit_at_5",
    "mrr",
    "p50_latency_ms",
    "p95_latency_ms",
]


def summarize_results(rows: list[dict], variants: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_rows": len(rows),
        "variants": {},
        "paired_comparisons": {},
        "diagnostics": {},
    }

    for variant in variants:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        diagnostic_counts = Counter(
            (row.get("diagnostic_result") or {}).get("category", "unknown")
            for row in variant_rows
        )
        rewrite_counts = Counter(row.get("rewrite_strategy") or "none" for row in variant_rows)

        file_hits = [row for row in variant_rows if (row.get("metrics") or {}).get("file_hit_at_5")]
        file_page_hit_count = 0
        for row in variant_rows:
            metrics = row.get("metrics") or {}
            if metrics.get("file_hit_at_5") and metrics.get("page_hit_at_5"):
                file_page_hit_count += 1
        file_page_hit_at_5 = (file_page_hit_count / len(variant_rows)) if variant_rows else None

        summary["variants"][variant] = {
            "rows": len(variant_rows),
            "hit_at_5": _average(variant_rows, "hit_at_5"),
            "initial_retrieval_hit_at_5": _average(variant_rows, "initial_retrieval_hit_at_5"),
            "final_retrieval_hit_at_5": _average(variant_rows, "final_retrieval_hit_at_5"),
            "file_hit_at_5": _average(variant_rows, "file_hit_at_5"),
            "file_page_hit_at_5": file_page_hit_at_5,
            "page_hit_at_5": _average(variant_rows, "page_hit_at_5"),
            "chunk_hit_at_5": _QRELS_NA,
            "root_hit_at_5": _QRELS_NA,
            "anchor_hit_at_5": _average(variant_rows, "anchor_hit_at_5"),
            "keyword_hit_at_5": _average(variant_rows, "keyword_hit_at_5"),
            "keyword_required_hit_at_5": _average(variant_rows, "keyword_required_hit_at_5"),
            "legacy_chunk_hit_at_5": _average(variant_rows, "legacy_chunk_hit_at_5"),
            "mrr": _average(variant_rows, "mrr"),
            "positive_chunk_mrr": _average(variant_rows, "positive_chunk_mrr"),
            "context_precision_id_at_5": _average(variant_rows, "context_precision_id_at_5"),
            "id_context_precision_at_5": _average(variant_rows, "id_context_precision_at_5"),
            "id_context_recall_at_5": _average(variant_rows, "id_context_recall_at_5"),
            "ndcg_at_5": _average(variant_rows, "ndcg_at_5"),
            "map_at_5": _average(variant_rows, "map_at_5"),
            "irrelevant_context_ratio_at_5": _average(variant_rows, "irrelevant_context_ratio_at_5"),
            "hard_negative_file_hit_at_5": _average(variant_rows, "hard_negative_file_hit_at_5"),
            "hard_negative_context_ratio_at_5": _average(variant_rows, "hard_negative_context_ratio_at_5"),
            "candidate_recall_before_rerank": _average(variant_rows, "candidate_recall_before_rerank"),
            "rerank_drop_rate": _average(variant_rows, "rerank_drop_rate"),
            "structure_drop_rate": _average(variant_rows, "structure_drop_rate"),
            "recall_at_5": _average(variant_rows, "recall_at_5"),
            "avg_latency_ms": _average_field(variant_rows, "latency_ms"),
            "p50_latency_ms": _percentile_field(variant_rows, "latency_ms", 0.50),
            "p95_latency_ms": _percentile_field(variant_rows, "latency_ms", 0.95),
            "retrieval_p50_ms": _percentile_field(variant_rows, "latency_ms", 0.50),
            "retrieval_p95_ms": _percentile_field(variant_rows, "latency_ms", 0.95),
            "error_rate": _average_field(variant_rows, "error_rate"),
            "fallback_trigger_rate": _average_field(variant_rows, "fallback_required"),
            "fallback_executed_rate": _average_field(variant_rows, "fallback_executed"),
            "fallback_helped_rate": _average(variant_rows, "fallback_helped"),
            "fallback_hurt_rate": _average(variant_rows, "fallback_hurt"),
            "rewrite_strategy_distribution": dict(rewrite_counts),
        }
        summary["diagnostics"][variant] = dict(diagnostic_counts)

    for pair_name, old_variant, new_variant in PAIR_DEFINITIONS:
        if old_variant in variants and new_variant in variants:
            summary["paired_comparisons"][pair_name] = _build_pairwise(rows, old_variant, new_variant)

    return summary


def render_summary_markdown(summary: dict) -> str:
    lines = [
        "# RAG Matrix Evaluation Summary",
        "",
        f"Generated at: `{summary.get('generated_at')}`",
        f"Rows: `{summary.get('sample_rows')}`",
        "",
        "## Variant Metrics",
        "",
        "| Variant | Rows | File@5 | File+Page@5 | CandRecall | HardNeg@5 | Anchor@5 | MRR | Chunk@5 | Root@5 | P50 ms | P95 ms | Error | FallbackReq | FallbackExec | Helped | Hurt |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, metrics in summary.get("variants", {}).items():
        lines.append(
            "| {variant} | {rows} | {file} | {file_page} | {candidate_recall} | {hard_neg} | {anchor} | {mrr} | {chunk} | {root} | {p50} | {p95} | {error} | {fallback} | {fallback_executed} | {fallback_helped} | {fallback_hurt} |".format(
                variant=variant,
                rows=metrics.get("rows", 0),
                file=_fmt_metric(metrics.get("file_hit_at_5")),
                file_page=_fmt_metric(metrics.get("file_page_hit_at_5")),
                candidate_recall=_fmt_metric(metrics.get("candidate_recall_before_rerank")),
                hard_neg=_fmt_metric(metrics.get("hard_negative_context_ratio_at_5")),
                anchor=_fmt_metric(metrics.get("anchor_hit_at_5")),
                mrr=_fmt_metric(metrics.get("mrr")),
                chunk=_fmt_metric(metrics.get("chunk_hit_at_5")),
                root=_fmt_metric(metrics.get("root_hit_at_5")),
                p50=_fmt_metric(metrics.get("p50_latency_ms")),
                p95=_fmt_metric(metrics.get("p95_latency_ms")),
                error=_fmt_metric(metrics.get("error_rate")),
                fallback=_fmt_metric(metrics.get("fallback_trigger_rate")),
                fallback_executed=_fmt_metric(metrics.get("fallback_executed_rate")),
                fallback_helped=_fmt_metric(metrics.get("fallback_helped_rate")),
                fallback_hurt=_fmt_metric(metrics.get("fallback_hurt_rate")),
            )
        )

    lines.extend(["", "## Paired Comparisons", ""])
    for pair_name, counts in summary.get("paired_comparisons", {}).items():
        lines.append(
            f"- `{pair_name}`: wins={counts.get('wins', 0)}, losses={counts.get('losses', 0)}, ties={counts.get('ties', 0)}, missing={counts.get('missing', 0)}, Chunk@5={_QRELS_NA}, Root@5={_QRELS_NA}"
        )

    lines.extend(["", "## Diagnostics", ""])
    for variant, counts in summary.get("diagnostics", {}).items():
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        lines.append(f"- `{variant}`: {rendered or 'none'}, Chunk@5={_QRELS_NA}, Root@5={_QRELS_NA}")

    lines.extend(["", "## Rewrite Strategies", ""])
    for variant, metrics in summary.get("variants", {}).items():
        counts = metrics.get("rewrite_strategy_distribution") or {}
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        lines.append(f"- `{variant}`: {rendered or 'none'}")

    lines.extend([
        "",
        "## Notes",
        "",
        "- `Chunk@5` and `Root@5` show `n/a (qrels missing)` because gold_chunk_ids are not available in the current dataset.",
        "- Primary metrics: `File@5`, `File+Page@5`, `CandidateRecall`, `HardNeg@5`, `Anchor@5`, `MRR`, `P50/P95`.",
        "- `HardNeg@5` is only meaningful on the gold dataset (hard_negative_files defined); shown as `-` on natural/frozen.",
        "- DeepEval/Ragas/TruLens are not runtime dependencies in this report.",
    ])
    return "\n".join(lines) + "\n"


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value if value == _QRELS_NA else str(value)[:12]
    if isinstance(value, bool):
        return "1.000" if value else "0.000"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _expected_fields(record: dict) -> dict[str, Any]:
    expected_chunk_ids = (
        _as_list(record.get("expected_chunk_ids"))
        or _as_list(record.get("legacy_gold_chunk_ids"))
        or _as_list(record.get("gold_chunk_ids"))
    )
    return {
        "expected_chunk_ids": expected_chunk_ids,
        "expected_root_ids": _as_list(record.get("expected_root_ids")),
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
        "score": doc.get("score"),
        "rerank_score": doc.get("rerank_score"),
        "final_score": doc.get("final_score"),
        "text_preview": text[:240],
    }


def _stage_hit(docs: list[dict], expected: dict[str, Any], top_k: int) -> bool:
    metrics = compute_retrieval_metrics(docs or [], top_k=top_k, **expected)
    return bool(metrics.get("hit_at_5") or metrics.get("id_context_recall_at_5"))


def _stage_metrics(meta: dict, expected: dict[str, Any], top_k: int) -> dict[str, Any]:
    before = meta.get("candidates_before_rerank") or []
    after_rerank = meta.get("candidates_after_rerank") or []
    after_structure = meta.get("candidates_after_structure_rerank") or []
    before_hit = _stage_hit(before, expected, top_k=max(top_k, len(before) or top_k))
    rerank_hit = _stage_hit(after_rerank, expected, top_k=max(top_k, len(after_rerank) or top_k))
    structure_hit = _stage_hit(after_structure, expected, top_k=top_k)
    return {
        "candidate_recall_before_rerank": 1.0 if before_hit else 0.0,
        "rerank_drop_rate": 1.0 if before_hit and not rerank_hit else 0.0,
        "structure_drop_rate": 1.0 if rerank_hit and not structure_hit else 0.0,
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


def _ensure_backend_imports() -> tuple[Any, Any, Any]:
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from rag_diagnostics import classify_failure
    from rag_pipeline import run_rag_graph
    from rag_utils import retrieve_documents

    return classify_failure, retrieve_documents, run_rag_graph


def evaluate_sample(record: dict, variant: str, top_k: int, mode: str = "retrieval") -> dict[str, Any]:
    classify_failure, retrieve_documents, run_rag_graph = _ensure_backend_imports()
    query = str(record.get("question") or record.get("query") or record.get("input") or "")
    sample_id = str(record.get("sample_id") or record.get("id") or query[:60])
    expected = _expected_fields(record)
    started = time.perf_counter()
    try:
        if mode == "graph":
            graph_result = run_rag_graph(query, context_files=None)
            meta = graph_result.get("rag_trace") or {}
            docs = graph_result.get("docs") or meta.get("retrieved_chunks") or []
            initial_docs = meta.get("initial_retrieved_chunks") or []
        else:
            retrieved = retrieve_documents(query, top_k=top_k, context_files=None)
            docs = retrieved.get("docs", [])
            meta = retrieved.get("meta", {})
            initial_docs = meta.get("candidates_after_structure_rerank") or docs

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
                "hit_at_5": False,
                "chunk_hit_at_5": False,
                "root_hit_at_5": False,
                "anchor_hit_at_5": False,
                "keyword_hit_at_5": False,
                "keyword_required_hit_at_5": False,
                "file_hit_at_5": False,
                "file_page_hit_at_5": False,
                "page_hit_at_5": False,
                "legacy_chunk_hit_at_5": False,
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
                "relevant_count": 0,
                "error_rate": 1.0,
            },
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


def _run_reindex(variant: str) -> None:
    print(f"reindex variant={variant} mode={VARIANT_CONFIGS[variant]['reindex_mode']}", flush=True)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "reindex_knowledge_base.py")],
        cwd=PROJECT_ROOT,
        env=_merged_env(variant),
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"reindex failed for {variant} with exit code {result.returncode}")


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


def _write_config(path: Path, args: argparse.Namespace, variants: list[str], destructive_reindex_run: bool) -> None:
    dataset_records = load_jsonl(args.dataset)
    dataset_validation = validate_eval_dataset_records(dataset_records, args.dataset)
    env_keys = [
        "EVAL_RETRIEVAL_TEXT_MODE",
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
    ]
    config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_schema_version": EVAL_SCHEMA_VERSION,
        "dataset_profile": args.dataset_profile,
        "dataset": str(args.dataset),
        "dataset_sha256": _sha256_file(args.dataset),
        "dataset_row_count": len(dataset_records),
        "dataset_schema_versions": _dataset_schema_versions(dataset_records),
        "dataset_validation": dataset_validation,
        "mode": args.mode,
        "limit": args.limit,
        "top_k": args.top_k,
        "variants": {variant: VARIANT_CONFIGS[variant] for variant in variants},
        "destructive_reindex_run": destructive_reindex_run,
        "skip_reindex": args.skip_reindex,
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
                    "rerank_drop_rate": metrics.get("rerank_drop_rate"),
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


def run_matrix(args: argparse.Namespace) -> int:
    variants = parse_variants(args.variants)
    report_dir = args.output_root / args.run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_eval_dataset_records(load_jsonl(args.dataset), args.dataset)
    if not validation["ok"]:
        (report_dir / "dataset_validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"dataset validation failed for {args.dataset}; "
            f"first errors: {validation['errors'][:3]}"
        )

    destructive_reindex_run = False
    variant_outputs: list[Path] = []
    last_reindex_mode = None
    for variant in variants:
        config = VARIANT_CONFIGS[variant]
        if config["requires_reindex"] and not args.skip_reindex:
            if not args.allow_destructive_reindex:
                raise RuntimeError(
                    f"{variant} requires destructive reindex. Pass --allow-destructive-reindex or --skip-reindex."
                )
            _run_reindex(variant)
            destructive_reindex_run = True
            last_reindex_mode = config["reindex_mode"]
        elif config["requires_reindex"] and args.skip_reindex:
            print(f"skip reindex variant={variant}", flush=True)
        elif last_reindex_mode and config["reindex_mode"] != last_reindex_mode:
            print(
                f"warning variant={variant} expects mode={config['reindex_mode']} but last reindex={last_reindex_mode}",
                flush=True,
            )

        variant_outputs.append(_run_worker(args.dataset, report_dir, variant, args.limit, args.top_k, args.mode))

    rows: list[dict] = []
    for output in variant_outputs:
        rows.extend(load_jsonl(output))

    write_jsonl(report_dir / "results.jsonl", rows)
    write_jsonl(report_dir / "miss_analysis.jsonl", _miss_analysis_rows(rows))
    summary = summarize_results(rows, variants=variants)
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (report_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    _write_config(report_dir / "config.json", args, variants, destructive_reindex_run)
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
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / ".jbeval" / "reports")
    parser.add_argument("--run-id", default=f"rag-matrix-{datetime.now().strftime('%Y%m%d-%H%M')}")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["retrieval", "graph"], default="retrieval")
    parser.add_argument("--variants", default="A0,A1,B1,G0,G1,G2,G3")
    parser.add_argument("--skip-reindex", action="store_true")
    parser.add_argument("--allow-destructive-reindex", action="store_true")
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
    args.output_root = args.output_root.resolve()
    if args.dataset_profile == "smoke" and args.limit is None:
        args.limit = 10

    if args.worker_variant:
        if args.worker_output is None:
            parser.error("--worker-output is required with --worker-variant")
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
