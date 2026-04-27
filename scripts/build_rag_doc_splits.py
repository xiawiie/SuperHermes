from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag_dataset_utils import (
    DEFAULT_DATASET,
    group_rows_by_source_file,
    load_jsonl,
    record_source_file,
    stable_digest,
    write_json,
    write_jsonl,
)


DEFAULT_SPLITS = PROJECT_ROOT / "eval" / "datasets" / "rag_doc_splits_v1.json"
DEFAULT_FROZEN = PROJECT_ROOT / "eval" / "datasets" / "rag_doc_frozen_eval_v1.jsonl"


def _file_sort_key(file_name: str) -> tuple[int, str]:
    return (int(stable_digest(file_name, 8), 16), file_name)


def assign_file_splits(rows: list[dict[str, Any]], dev_files: int = 9, test_files: int = 12) -> dict[str, Any]:
    grouped = group_rows_by_source_file(rows)
    files = sorted(grouped, key=_file_sort_key)
    total = len(files)
    test_count = max(0, min(test_files, total))
    dev_count = max(0, min(dev_files, total - test_count))
    test = sorted(files[:test_count])
    dev = sorted(files[test_count : test_count + dev_count])
    train = sorted(files[test_count + dev_count :])
    return {
        "schema_version": "rag-doc-splits-v1",
        "source_dataset": str(DEFAULT_DATASET),
        "split_key": "gold_files[0]",
        "row_count": len(rows),
        "source_file_count": total,
        "splits": {"train": train, "dev": dev, "test": test},
        "answer_type_counts": dict(Counter(str(row.get("answer_type") or "") for row in rows)),
    }


def split_for_file(file_name: str, manifest: dict[str, Any]) -> str:
    for split, files in (manifest.get("splits") or {}).items():
        if file_name in files:
            return split
    return ""


def derive_frozen_eval_rows(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    frozen: list[dict[str, Any]] = []
    for row in rows:
        split = split_for_file(record_source_file(row), manifest)
        if split in {"dev", "test"}:
            out = dict(row)
            out["split"] = split
            frozen.append(out)
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(description="Build source-file grouped RAG dataset splits.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--splits-output", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--frozen-output", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--dev-files", type=int, default=9)
    parser.add_argument("--test-files", type=int, default=12)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    manifest = assign_file_splits(rows, dev_files=args.dev_files, test_files=args.test_files)
    manifest["source_dataset"] = str(args.input)
    write_json(args.splits_output, manifest)
    write_jsonl(args.frozen_output, derive_frozen_eval_rows(rows, manifest))
    print(f"Wrote split manifest to {args.splits_output}")
    print(f"Wrote frozen eval rows to {args.frozen_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
