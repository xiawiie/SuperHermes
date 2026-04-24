from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag_dataset_utils import as_list, load_jsonl, loose_text, write_json, write_jsonl


DEFAULT_INPUT = PROJECT_ROOT / ".jbeval" / "datasets" / "rag_chunk_gold_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / ".jbeval" / "datasets" / "rag_contrastive_train_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / ".jbeval" / "datasets" / "rag_contrastive_train_v1.report.json"


def _chunk_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_id") or "")


def _file_name(chunk: dict[str, Any]) -> str:
    return str(chunk.get("filename") or chunk.get("file_name") or "")


def _context(chunk: dict[str, Any], negative_type: str = "", relevance: int = 0) -> dict[str, Any]:
    return {
        "chunk_id": _chunk_id(chunk),
        "root_chunk_id": str(chunk.get("root_chunk_id") or ""),
        "file_name": _file_name(chunk),
        "page_number": int(chunk.get("page_number") or 0),
        "text": str(chunk.get("text") or chunk.get("anchor_text") or "")[:1200],
        "negative_type": negative_type,
        "relevance": relevance,
    }


def _shares_query_signal(row: dict[str, Any], chunk: dict[str, Any]) -> bool:
    haystack = loose_text(" ".join([str(chunk.get("text") or ""), str(chunk.get("retrieval_text") or ""), _file_name(chunk)]))
    needles = [loose_text(row.get("query") or row.get("question") or "")]
    needles.extend(loose_text(item) for item in as_list(row.get("expected_keywords")) if loose_text(item))
    return any(needle and (needle in haystack or haystack in needle) for needle in needles)


def build_contrastive_row(
    row: dict[str, Any],
    chunks: list[dict[str, Any]],
    min_hard: int = 2,
    min_easy: int = 1,
) -> dict[str, Any]:
    positive_ids = {str(item) for item in as_list(row.get("gold_chunk_ids")) if str(item)}
    positive_roots = {str(item) for item in as_list(row.get("expected_root_ids")) if str(item)}
    hard_files = {str(item) for item in as_list(row.get("hard_negative_files")) if str(item)}
    positives = list(row.get("supporting_chunks") or [])

    hard: list[dict[str, Any]] = []
    easy: list[dict[str, Any]] = []
    for chunk in chunks:
        cid = _chunk_id(chunk)
        root = str(chunk.get("root_chunk_id") or "")
        if not cid or cid in positive_ids or root in positive_roots:
            continue
        if _file_name(chunk) in hard_files or _shares_query_signal(row, chunk):
            hard.append(_context(chunk, "hard_negative_file" if _file_name(chunk) in hard_files else "lexical_near_miss", 0))
        else:
            easy.append(_context(chunk, "random_easy", 0))

    return {
        "id": f"contrastive_{row.get('id', '')}",
        "schema_version": "rag-contrastive-train-v1",
        "split": row.get("split", ""),
        "query": row.get("query") or row.get("question") or "",
        "positive_contexts": positives,
        "hard_negatives": hard[:min_hard],
        "easy_negatives": easy[:min_easy],
        "labels": {
            "positive_relevance": 3,
            "hard_negative_relevance": 0,
            "easy_negative_relevance": 0,
        },
        "provenance": {"source_chunk_gold_id": row.get("id", "")},
    }


def build_contrastive_rows(rows: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        build_contrastive_row(row, chunks)
        for row in rows
        if row.get("split") == "train" and (row.get("quality") or {}).get("alignment_status") == "aligned"
    ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hard_counts = Counter(len(row.get("hard_negatives") or []) for row in rows)
    easy_counts = Counter(len(row.get("easy_negatives") or []) for row in rows)
    return {
        "schema_version": "rag-contrastive-report-v1",
        "row_count": len(rows),
        "hard_negative_count_distribution": dict(hard_counts),
        "easy_negative_count_distribution": dict(easy_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine contrastive negatives for RAG training rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--chunks-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    chunks = load_jsonl(args.chunks_jsonl)
    out = build_contrastive_rows(rows, chunks)
    write_jsonl(args.output, out)
    write_json(args.report, summarize(out))
    print(f"Wrote {len(out)} contrastive rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
