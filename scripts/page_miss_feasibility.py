from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _bucket(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") or {}
    file_recall = metrics.get("file_candidate_recall_before_rerank")
    file_hit = metrics.get("file_hit_at_5")
    file_page_hit = metrics.get("file_page_hit_at_5")
    chunk_hit = metrics.get("chunk_hit_at_5")
    root_hit = metrics.get("root_hit_at_5")
    hard_neg_ratio = metrics.get("hard_negative_context_ratio_at_5")
    page_before = metrics.get("page_rank_before_rerank")
    page_after = metrics.get("page_rank_after_rerank")

    if isinstance(file_recall, (int, float)) and file_recall <= 0:
        return "file_missing"

    if page_before is not None and (page_after is None or (isinstance(page_after, int) and page_after > 5)):
        return "page_hit_before_rerank_dropped"

    if bool(root_hit) and not bool(chunk_hit):
        return "root_hit_chunk_wrong"

    if bool(file_hit) and not bool(file_page_hit):
        return "file_hit_page_missing"

    if isinstance(hard_neg_ratio, (int, float)) and hard_neg_ratio > 0 and not bool(file_page_hit):
        return "hard_negative_page_confusion"

    return "other"


def analyze(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    variant_rows = [r for r in rows if r.get("variant") == variant]
    page_miss_rows = [r for r in variant_rows if not bool((r.get("metrics") or {}).get("file_page_hit_at_5"))]
    buckets = Counter(_bucket(r) for r in page_miss_rows)
    total_page_miss = len(page_miss_rows)

    target = buckets.get("page_hit_before_rerank_dropped", 0) + buckets.get("root_hit_chunk_wrong", 0)
    target_ratio = (target / total_page_miss) if total_page_miss else 0.0
    gs3p_feasible = target_ratio >= 0.50

    return {
        "variant": variant,
        "rows": len(variant_rows),
        "page_miss_rows": total_page_miss,
        "buckets": dict(buckets),
        "target_ratio": round(target_ratio, 4),
        "gs3p_feasible": gs3p_feasible,
        "decision_rule": "page_hit_before_rerank_dropped + root_hit_chunk_wrong >= 50% of page_miss",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Page Miss Feasibility Analysis",
        "",
        f"- Variant: `{report['variant']}`",
        f"- Rows: `{report['rows']}`",
        f"- Page miss rows: `{report['page_miss_rows']}`",
        f"- Target ratio: `{report['target_ratio']:.4f}`",
        f"- GS3P feasible: `{report['gs3p_feasible']}`",
        "",
        "## Buckets",
        "",
        "| Bucket | Count |",
        "| --- | ---: |",
    ]
    for k, v in sorted((report.get("buckets") or {}).items()):
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "## Rule",
            "",
            f"- {report['decision_rule']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze page miss feasibility for GS3P.")
    parser.add_argument("--results", type=Path, required=True, help="Path to results.jsonl")
    parser.add_argument("--variant", default="GS3")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    report = analyze(rows, args.variant)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "page_miss_feasibility.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "page_miss_feasibility.md").write_text(render_md(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
