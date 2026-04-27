from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANCHOR_PATTERN = re.compile(
    r"(第[一二三四五六七八九十百千万零两0-9]+[编章节条部分款项]|"
    r"\d+(?:\.\d+){1,4}|"
    r"[一二三四五六七八九十]+、|"
    r"[（(][一二三四五六七八九十0-9A-Za-z]+[)）]|"
    r"附录[A-Za-z0-9一二三四五六七八九十]+|"
    r"附件[0-9一二三四五六七八九十]+)"
)


def _extract_anchors(*values: object) -> list[str]:
    anchors: list[str] = []
    for value in values:
        if isinstance(value, list):
            text = " ".join(str(item) for item in value)
        else:
            text = str(value or "")
        anchors.extend(ANCHOR_PATTERN.findall(text))
    return list(dict.fromkeys(item for item in anchors if item))


def derive_record(record: dict) -> dict:
    derived = dict(record)
    expected_keywords = list(record.get("expected_keywords") or [])
    anchors = _extract_anchors(
        record.get("question"),
        record.get("expected_answer"),
        expected_keywords,
    )
    derived["expected_anchors"] = anchors
    derived.setdefault("expected_root_ids", [])
    derived["expected_keywords"] = expected_keywords
    derived["legacy_gold_chunk_ids"] = list(record.get("gold_chunk_ids") or [])
    return derived


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive structure-aware RAG evaluation expectations.")
    parser.add_argument("input", type=Path, help="Source rag_tuning.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/datasets/rag_tuning_derived.jsonl"),
        help="Output JSONL path",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            dst.write(json.dumps(derive_record(record), ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
