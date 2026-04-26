from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _category(row: dict[str, Any]) -> str:
    metrics = row.get("metrics") or {}
    if metrics.get("file_hit_at_5") and metrics.get("file_page_hit_at_5") and metrics.get("chunk_hit_at_5"):
        return "all_hit"
    if metrics.get("file_hit_at_5") and not metrics.get("file_page_hit_at_5"):
        return "page_miss"
    if metrics.get("file_hit_at_5") and metrics.get("file_page_hit_at_5") and not metrics.get("chunk_hit_at_5"):
        return "chunk_miss"
    if not metrics.get("file_hit_at_5"):
        return "file_miss"
    return "other"


def _trace_entry(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    trace = row.get("trace") or {}
    return {
        "sample_id": row.get("sample_id"),
        "variant": row.get("variant"),
        "query": row.get("query"),
        "category": _category(row),
        "expected": row.get("expected"),
        "metrics": {
            "file_hit_at_5": metrics.get("file_hit_at_5"),
            "file_page_hit_at_5": metrics.get("file_page_hit_at_5"),
            "chunk_hit_at_5": metrics.get("chunk_hit_at_5"),
            "root_hit_at_5": metrics.get("root_hit_at_5"),
            "candidate_recall_before_rerank": metrics.get("candidate_recall_before_rerank"),
            "file_candidate_recall_before_rerank": metrics.get("file_candidate_recall_before_rerank"),
            "page_rank_before_rerank": metrics.get("page_rank_before_rerank"),
            "page_rank_after_rerank": metrics.get("page_rank_after_rerank"),
            "page_rank_delta": metrics.get("page_rank_delta"),
        },
        "query_plan": trace.get("query_plan"),
        "pre_rerank_candidates": trace.get("candidates_before_rerank", []),
        "post_rerank_candidates": trace.get("candidates_after_rerank", []),
        "final_context": row.get("retrieved_chunks", []),
    }


def select_samples(rows: list[dict[str, Any]], variants: list[str], sample_count: int = 12) -> list[str]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if r.get("variant") in variants:
            by_sample[str(r.get("sample_id"))][str(r.get("variant"))] = r

    scored: list[tuple[int, str]] = []
    for sid, mapping in by_sample.items():
        if not all(v in mapping for v in variants):
            continue
        # prioritize disagreement and harder cases
        hits = []
        for v in variants:
            m = mapping[v].get("metrics") or {}
            hits.append(1 if m.get("file_page_hit_at_5") else 0)
        disagreement = len(set(hits)) > 1
        hard = any(_category(mapping[v]) in {"file_miss", "page_miss", "chunk_miss"} for v in variants)
        score = (2 if disagreement else 0) + (1 if hard else 0)
        scored.append((score, sid))

    scored.sort(key=lambda x: (-x[0], x[1]))
    selected = [sid for _, sid in scored[:sample_count]]
    return selected


def analyze(rows: list[dict[str, Any]], variants: list[str], sample_count: int) -> dict[str, Any]:
    selected = select_samples(rows, variants, sample_count)
    by_sample_variant = {(str(r.get("sample_id")), str(r.get("variant"))): r for r in rows}
    traces: list[dict[str, Any]] = []
    category_counts: dict[str, int] = defaultdict(int)

    for sid in selected:
        for variant in variants:
            row = by_sample_variant.get((sid, variant))
            if not row:
                continue
            entry = _trace_entry(row)
            traces.append(entry)
            category_counts[entry["category"]] += 1

    return {
        "variants": variants,
        "selected_sample_ids": selected,
        "trace_count": len(traces),
        "category_counts": dict(category_counts),
        "traces": traces,
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Golden Trace Analysis",
        "",
        f"- Variants: `{', '.join(report.get('variants') or [])}`",
        f"- Selected samples: `{len(report.get('selected_sample_ids') or [])}`",
        f"- Trace count: `{report.get('trace_count', 0)}`",
        "",
        "## Category Counts",
        "",
        "| Category | Count |",
        "| --- | ---: |",
    ]
    for k, v in sorted((report.get("category_counts") or {}).items()):
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "## Selected Samples", ""])
    for sid in report.get("selected_sample_ids") or []:
        lines.append(f"- `{sid}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build golden trace analysis from results.jsonl")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--variants", default="GS3,V3Q,GS2HR")
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    rows = load_jsonl(args.results)
    report = analyze(rows, variants, args.sample_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "golden_trace_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "golden_trace_analysis.md").write_text(render_md(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_samples": len(report.get("selected_sample_ids") or []),
                "trace_count": report.get("trace_count", 0),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
