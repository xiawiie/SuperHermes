from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_rag_doc_splits import split_for_file
from scripts.rag_dataset_utils import (
    DEFAULT_DATASET,
    alignment_score,
    load_jsonl,
    record_source_file,
    write_json,
    write_jsonl,
)
from scripts.rag_qrels import (
    DEFAULT_ROOT_GRANULARITY,
    DEFAULT_ROOT_TYPE,
    QREL_SCHEMA_VERSION,
    attach_canonical_ids,
    qrel_report_output_paths,
)


DEFAULT_SPLITS = PROJECT_ROOT / "eval" / "datasets" / "rag_doc_splits_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "datasets" / "rag_chunk_gold_v2.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "eval" / "datasets" / "rag_chunk_gold_v2.report.json"
MIN_ALIGNMENT_SCORE = 0.75
AMBIGUOUS_ALIGNMENT_SCORE = 0.55


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _chunk_file(chunk: dict[str, Any]) -> str:
    return str(chunk.get("filename") or chunk.get("file_name") or "")


def _candidate_chunks(row: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_file = record_source_file(row)
    return [chunk for chunk in chunks if _chunk_file(chunk) == source_file]


def _supporting_chunk(chunk: dict[str, Any], score: float, method: str) -> dict[str, Any]:
    chunk = attach_canonical_ids(chunk)
    page_number = int(chunk.get("page_number") or 0)
    return {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "root_chunk_id": str(chunk.get("root_chunk_id") or ""),
        "parent_chunk_id": str(chunk.get("parent_chunk_id") or ""),
        "canonical_chunk_id": str(chunk.get("canonical_chunk_id") or ""),
        "canonical_root_id": str(chunk.get("canonical_root_id") or ""),
        "file_name": _chunk_file(chunk),
        "page_number": page_number,
        "page_start": int(chunk.get("page_start") or page_number),
        "page_end": int(chunk.get("page_end") or page_number),
        "anchor_id": str(chunk.get("anchor_id") or ""),
        "section_title": str(chunk.get("section_title") or ""),
        "section_path": str(chunk.get("section_path") or ""),
        "root_type": DEFAULT_ROOT_TYPE,
        "root_granularity": DEFAULT_ROOT_GRANULARITY,
        "anchor_text": str(chunk.get("text") or "")[:500],
        "match_method": method,
        "match_score": round(score, 4),
        "relevance": 3,
    }


def align_row(row: dict[str, Any], chunks: list[dict[str, Any]], split: str) -> dict[str, Any]:
    candidates = _candidate_chunks(row, chunks)
    scored = sorted(
        ((alignment_score(row, chunk), chunk) for chunk in candidates),
        key=lambda item: item[0].score,
        reverse=True,
    )
    selected = [(score, chunk) for score, chunk in scored if score.score >= MIN_ALIGNMENT_SCORE][:3]
    ambiguous = [(score, chunk) for score, chunk in scored if AMBIGUOUS_ALIGNMENT_SCORE <= score.score < MIN_ALIGNMENT_SCORE][:5]
    out = dict(row)
    out["schema_version"] = QREL_SCHEMA_VERSION
    out["source_schema_version"] = row.get("benchmark_schema_version") or "rag-doc-gold-v2"
    out["split"] = split
    out["task_type"] = row.get("answer_type") or (row.get("metadata") or {}).get("answer_type") or ""
    out["reference_answer_extract"] = row.get("reference_answer") or row.get("expected_answer") or ""
    out.setdefault("reference_answer_abstractive", "")
    out["root_type"] = DEFAULT_ROOT_TYPE
    out["root_granularity"] = DEFAULT_ROOT_GRANULARITY

    if selected:
        supporting = [_supporting_chunk(chunk, score.score, score.method) for score, chunk in selected]
        out["gold_chunk_ids"] = [item["chunk_id"] for item in supporting if item["chunk_id"]]
        out["expected_root_ids"] = sorted({item["root_chunk_id"] for item in supporting if item["root_chunk_id"]})
        out["canonical_chunk_ids"] = [item["canonical_chunk_id"] for item in supporting if item["canonical_chunk_id"]]
        out["canonical_root_ids"] = sorted({item["canonical_root_id"] for item in supporting if item["canonical_root_id"]})
        out["supporting_chunks"] = supporting
        out["quality"] = {
            "alignment_status": "aligned",
            "alignment_confidence": supporting[0]["match_score"],
            "quality_score": (row.get("quality_checks") or {}).get("quality_score", 0.0),
            "review_status": "draft",
            "evidence_type": "direct_answer",
            "rejection_reason": "",
            "reviewer_notes": "",
        }
    elif ambiguous:
        supporting = [_supporting_chunk(chunk, score.score, score.method) for score, chunk in ambiguous]
        out["gold_chunk_ids"] = []
        out["expected_root_ids"] = []
        out["canonical_chunk_ids"] = []
        out["canonical_root_ids"] = []
        out["supporting_chunks"] = supporting
        out["quality"] = {
            "alignment_status": "ambiguous",
            "alignment_failure_reason": "ambiguous_low_confidence",
            "alignment_confidence": supporting[0]["match_score"],
            "quality_score": (row.get("quality_checks") or {}).get("quality_score", 0.0),
            "review_status": "needs_review",
            "evidence_type": "",
            "rejection_reason": "",
            "reviewer_notes": "",
        }
    else:
        out["gold_chunk_ids"] = []
        out["expected_root_ids"] = []
        out["canonical_chunk_ids"] = []
        out["canonical_root_ids"] = []
        out["supporting_chunks"] = []
        out["quality"] = {
            "alignment_status": "failed",
            "alignment_failure_reason": "no_candidate_same_file" if not candidates else "low_alignment_score",
            "alignment_confidence": round(scored[0][0].score, 4) if scored else 0.0,
            "quality_score": (row.get("quality_checks") or {}).get("quality_score", 0.0),
            "review_status": "draft",
            "evidence_type": "",
            "rejection_reason": "",
            "reviewer_notes": "",
        }

    out["provenance"] = {
        "source_dataset": str(DEFAULT_DATASET),
        "source_id": row.get("id", ""),
        "index_profile": "gold_tcf",
        "collection": "embeddings_collection_gold_tcf",
        "index_variant": "title_context_filename",
        "chunker_version": "document_loader_current",
        "qrel_id_policy": "collection_local_and_canonical",
    }
    return out


def build_chunk_gold_rows(
    rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        split = split_for_file(record_source_file(row), manifest)
        out.append(align_row(row, chunks, split=split))
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter((row.get("quality") or {}).get("alignment_status") for row in rows)
    reviews = Counter((row.get("quality") or {}).get("review_status") for row in rows)
    split_counts = Counter(row.get("split") for row in rows)
    failures = Counter((row.get("quality") or {}).get("alignment_failure_reason") for row in rows if (row.get("quality") or {}).get("alignment_failure_reason"))
    chunk_qrel_rows = sum(1 for row in rows if row.get("gold_chunk_ids") or row.get("canonical_chunk_ids"))
    root_qrel_rows = sum(1 for row in rows if row.get("expected_root_ids") or row.get("canonical_root_ids"))
    return {
        "schema_version": "rag-chunk-gold-report-v2",
        "row_count": len(rows),
        "split_counts": dict(split_counts),
        "alignment_status_counts": dict(statuses),
        "review_status_counts": dict(reviews),
        "alignment_failure_reasons": dict(failures),
        "aligned_rate": round(statuses.get("aligned", 0) / max(1, len(rows)), 4),
        "chunk_qrel_rows": chunk_qrel_rows,
        "root_qrel_rows": root_qrel_rows,
        "chunk_qrel_coverage": round(chunk_qrel_rows / max(1, len(rows)), 4),
        "root_qrel_coverage": round(root_qrel_rows / max(1, len(rows)), 4),
        "root_type": DEFAULT_ROOT_TYPE,
        "root_granularity": DEFAULT_ROOT_GRANULARITY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Align page-level RAG gold rows to chunk ids.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--chunks-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write-review", action="store_true")
    parser.add_argument("--write-samples", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    manifest = _load_json(args.splits)
    chunks = load_jsonl(args.chunks_jsonl)
    out = build_chunk_gold_rows(rows, chunks, manifest)
    write_jsonl(args.output, out)
    write_json(args.report, summarize(out))
    if args.write_review or args.write_samples:
        paths = qrel_report_output_paths(args.output.with_suffix(""))
        if args.write_review:
            write_jsonl(paths["review"], out)
        if args.write_samples:
            failed = [row for row in out if (row.get("quality") or {}).get("alignment_status") == "failed"][:50]
            ambiguous = [row for row in out if (row.get("quality") or {}).get("alignment_status") == "ambiguous"][:50]
            write_jsonl(paths["failed_sample"], failed)
            write_jsonl(paths["ambiguous_sample"], ambiguous)
    print(f"Wrote {len(out)} chunk gold rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
