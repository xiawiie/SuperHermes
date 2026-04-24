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

from scripts.rag_dataset_utils import load_jsonl, write_json


def _result(errors: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {"ok": not errors, "errors": sorted(set(errors)), "details": details}


def validate_split_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    splits = manifest.get("splits") or {}
    seen: dict[str, str] = {}
    overlaps: list[str] = []
    for split, files in splits.items():
        for file_name in files:
            if file_name in seen:
                overlaps.append(str(file_name))
            seen[str(file_name)] = str(split)
    if overlaps:
        errors.append("split_file_overlap")
    return _result(errors, {"source_file_count": len(seen), "overlaps": sorted(set(overlaps))})


def validate_chunk_gold_rows(rows: list[dict[str, Any]], min_alignment_rate: float = 0.9) -> dict[str, Any]:
    errors: list[str] = []
    statuses = Counter((row.get("quality") or {}).get("alignment_status") for row in rows)
    for row in rows:
        if (row.get("quality") or {}).get("alignment_status") == "aligned" and not row.get("supporting_chunks"):
            errors.append("aligned_row_without_supporting_chunks")
        if row.get("supporting_chunks") and not row.get("gold_chunk_ids"):
            errors.append("supporting_chunks_without_gold_chunk_ids")
    aligned_rate = statuses.get("aligned", 0) / max(1, len(rows))
    if rows and aligned_rate < min_alignment_rate:
        errors.append("alignment_rate_below_threshold")
    return _result(
        errors,
        {
            "row_count": len(rows),
            "alignment_status_counts": dict(statuses),
            "aligned_rate": round(aligned_rate, 4),
        },
    )


def validate_contrastive_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    for row in rows:
        positives = {ctx.get("chunk_id") for ctx in row.get("positive_contexts") or []}
        negatives = {ctx.get("chunk_id") for ctx in (row.get("hard_negatives") or []) + (row.get("easy_negatives") or [])}
        if positives & negatives:
            errors.append("positive_negative_collision")
        if not positives:
            errors.append("missing_positive_contexts")
        if not row.get("hard_negatives"):
            errors.append("missing_hard_negatives")
    return _result(errors, {"row_count": len(rows)})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate derived RAG dataset artifacts.")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--chunk-gold", type=Path)
    parser.add_argument("--contrastive", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {"schema_version": "rag-dataset-validation-v1"}
    if args.splits:
        report["splits"] = validate_split_manifest(json.loads(args.splits.read_text(encoding="utf-8")))
    if args.chunk_gold:
        report["chunk_gold"] = validate_chunk_gold_rows(load_jsonl(args.chunk_gold))
    if args.contrastive:
        report["contrastive"] = validate_contrastive_rows(load_jsonl(args.contrastive))
    report["ok"] = all(section.get("ok", True) for section in report.values() if isinstance(section, dict))
    write_json(args.report, report)
    print(f"Wrote validation report to {args.report}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
