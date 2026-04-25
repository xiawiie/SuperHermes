"""Analyze RAG retrieval misses from evaluation results.

Five-category miss diagnosis:
  - file_recall_miss: correct file not in candidates at all
  - page_miss: correct file present but wrong page
  - ranking_miss: correct candidate exists but rerank dropped it out of top5
  - hard_negative_confusion: all top5 from hard negative files
  - low_confidence: top1 score below threshold

Plus: CandRecall bucketing, fuzzy match histogram, family confusion,
rerank drop list, false-retrieval top10.

Usage:
    python scripts/analyze_rag_misses.py --input .jbeval/reports/REPORT/miss_analysis.jsonl
    python scripts/analyze_rag_misses.py --input .jbeval/reports/REPORT/results.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_TOP1_LOW_CONFIDENCE_THRESHOLD = 0.20
_FAMILY_PREFIX_LEN = 6
_FUZZY_HIST_BINS = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]


def _extract_book_title(query: str) -> str:
    """Extract content inside 《》from a query."""
    m = re.search(r"《([^》]+)》", query)
    return m.group(1) if m else ""


def _normalize_filename(name: str) -> str:
    """Strip extension and common suffixes for fuzzy matching."""
    import os as _os
    base = _os.path.splitext(name)[0]
    base = re.sub(r"[_\s]*副本$", "", base)
    base = re.sub(r"\(\d+\)$", "", base)
    base = re.sub(r"（[^）]*）$", "", base)
    base = base.lower().strip()
    return base


def _page_candidates(doc: dict) -> set[str]:
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
        for page in range(page_start, min(page_end, page_start + 20) + 1):
            pages.add(str(page))
            pages.add(str(page + 1))
    elif page_number is None and page_start is None and page_end is None:
        return pages

    return pages


def _family_prefix(filename: str) -> str:
    """Extract product family prefix for confusion analysis."""
    norm = _normalize_filename(filename)
    parts = norm.split()
    if parts:
        return parts[0][:_FAMILY_PREFIX_LEN].upper()
    return norm[:_FAMILY_PREFIX_LEN].upper()


def _fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _classify_miss(
    row: dict,
    results: list[dict] | None = None,
) -> str:
    """Classify a miss into one of five categories."""
    metrics = row.get("metrics") or {}
    trace = row.get("trace") or {}
    expected = row.get("expected") or {}

    expected_files = set(expected.get("expected_files") or [])
    hard_neg_files = set(expected.get("hard_negative_files") or [])

    # Get candidate and top5 file info
    stage_candidates = row.get("stage_candidates") or {}
    candidates_before = (
        trace.get("candidates_before_rerank")
        or stage_candidates.get("before_rerank")
        or []
    )
    candidates_known = (
        "candidates_before_rerank" in trace
        or "before_rerank" in stage_candidates
    )
    top5_chunks = row.get("retrieved_chunks") or []
    top5_files = {str(c.get("filename") or "") for c in top5_chunks}
    candidate_files = {str(c.get("filename") or "") for c in candidates_before}

    # hard_negative_confusion: all top5 from hard negatives
    if hard_neg_files and top5_files and top5_files.issubset(hard_neg_files):
        return "hard_negative_confusion"

    # file_recall_miss: correct file not in candidates at all
    if expected_files and candidates_known and not (expected_files & candidate_files):
        return "file_recall_miss"

    # ranking_miss: correct file in candidates but not in top5
    cand_recall = metrics.get("candidate_recall_before_rerank")
    if expected_files and candidates_known and (expected_files & candidate_files) and not (expected_files & top5_files):
        return "ranking_miss"

    # page_miss: correct file in top5 but page doesn't match
    if expected_files and (expected_files & top5_files):
        expected_pages = set(str(p) for p in (expected.get("expected_pages") or []))
        if expected_pages:
            top5_pages = set()
            for c in top5_chunks:
                if str(c.get("filename") or "") in expected_files:
                    top5_pages |= _page_candidates(c)
            if not (expected_pages & top5_pages):
                return "page_miss"
            return "ok"
        return "ok"

    # low_confidence: top1 score is very low
    top1_score = None
    for c in top5_chunks:
        s = c.get("score") or c.get("rerank_score") or c.get("final_score")
        if s is not None:
            top1_score = float(s)
            break
    if top1_score is not None and top1_score < _TOP1_LOW_CONFIDENCE_THRESHOLD:
        return "low_confidence"

    return "file_recall_miss"


def _candrecall_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    v = float(value)
    if v == 0:
        return "0"
    if v < 0.5:
        return "0<x<0.5"
    return ">=0.5"


def _fuzzy_bin(ratio: float) -> str:
    for lo, hi in _FUZZY_HIST_BINS:
        if lo <= ratio < hi:
            return f"{lo:.1f}-{hi:.1f}"
    return "1.0"


def analyze_misses(miss_rows: list[dict], results_rows: list[dict] | None = None) -> dict[str, Any]:
    """Perform comprehensive miss analysis on evaluation miss rows."""
    report: dict[str, Any] = {}

    # 1. Five-category classification
    category_counts = Counter()
    category_examples: dict[str, list[str]] = defaultdict(list)
    for row in miss_rows:
        cat = _classify_miss(row, results=results_rows)
        category_counts[cat] += 1
        if len(category_examples[cat]) < 5:
            category_examples[cat].append(str(row.get("query") or row.get("sample_id") or ""))

    report["category_counts"] = dict(category_counts)
    report["category_examples"] = dict(category_examples)
    total = sum(category_counts.values())
    if total > 0:
        report["category_pcts"] = {k: round(v / total, 3) for k, v in category_counts.items()}

    # 2. CandRecall bucketing
    cand_buckets = Counter()
    for row in miss_rows:
        metrics = row.get("metrics") or {}
        cr = metrics.get("candidate_recall_before_rerank")
        cand_buckets[_candrecall_bucket(cr)] += 1
    report["cand_recall_buckets"] = dict(cand_buckets)

    # 3. Fuzzy match histogram: 《》content vs filename
    fuzzy_hist = Counter()
    fuzzy_details: list[dict] = []
    for row in miss_rows:
        query = str(row.get("query") or "")
        book_title = _extract_book_title(query)
        if not book_title:
            continue
        expected = row.get("expected") or {}
        expected_files = expected.get("expected_files") or []
        if not expected_files:
            continue
        target_file = str(expected_files[0])
        norm_title = _normalize_filename(book_title)
        norm_file = _normalize_filename(target_file)
        ratio = _fuzzy_ratio(norm_title, norm_file)
        fuzzy_hist[_fuzzy_bin(ratio)] += 1
        if len(fuzzy_details) < 20:
            fuzzy_details.append({
                "book_title": book_title,
                "target_file": target_file,
                "ratio": round(ratio, 3),
            })
    report["fuzzy_histogram"] = dict(sorted(fuzzy_hist.items()))
    report["fuzzy_details_sample"] = fuzzy_details

    # 4. Hard negative family confusion
    family_confusion: dict[str, Counter] = defaultdict(Counter)
    for row in miss_rows:
        top5_chunks = row.get("retrieved_chunks") or []
        expected = row.get("expected") or {}
        expected_files = expected.get("expected_files") or []
        if not expected_files:
            continue
        expected_family = _family_prefix(str(expected_files[0]))
        for c in top5_chunks:
            f = str(c.get("filename") or "")
            retrieved_family = _family_prefix(f)
            if retrieved_family != expected_family:
                family_confusion[expected_family][retrieved_family] += 1
    report["family_confusion"] = {k: dict(v.most_common(5)) for k, v in family_confusion.items()}

    # 5. Rerank drop top20
    rerank_drops: list[dict] = []
    for row in miss_rows:
        metrics = row.get("metrics") or {}
        if metrics.get("rerank_drop_rate"):
            rerank_drops.append({
                "sample_id": row.get("sample_id"),
                "query": str(row.get("query") or "")[:80],
            })
    report["rerank_drop_count"] = len(rerank_drops)
    report["rerank_drop_top20"] = rerank_drops[:20]

    # 6. False-retrieval (mistakenly retrieved) top10 files
    false_file_counts = Counter()
    for row in miss_rows:
        top5_chunks = row.get("retrieved_chunks") or []
        expected = row.get("expected") or {}
        expected_files = set(expected.get("expected_files") or [])
        for c in top5_chunks:
            f = str(c.get("filename") or "")
            if f and f not in expected_files:
                false_file_counts[f] += 1
    report["false_retrieval_top10"] = dict(false_file_counts.most_common(10))

    # 7. Anchor hit rate
    anchor_hits = 0
    anchor_total = 0
    for row in miss_rows:
        metrics = row.get("metrics") or {}
        if metrics.get("anchor_hit_at_5") is not None:
            anchor_total += 1
            if metrics.get("anchor_hit_at_5"):
                anchor_hits += 1
    if anchor_total:
        report["anchor_hit_rate"] = round(anchor_hits / anchor_total, 3)

    return report


def _render_report(report: dict) -> str:
    lines = ["# RAG Miss Analysis Report", ""]

    lines.append("## Category Distribution")
    total = sum(report.get("category_counts", {}).values())
    for cat, count in sorted(report.get("category_counts", {}).items(), key=lambda x: -x[1]):
        pct = report.get("category_pcts", {}).get(cat, 0)
        lines.append(f"- `{cat}`: {count} ({pct:.1%})")
    lines.append("")

    lines.append("## CandRecall Buckets")
    for bucket, count in sorted(report.get("cand_recall_buckets", {}).items()):
        lines.append(f"- `{bucket}`: {count}")
    lines.append("")

    lines.append("## Fuzzy Match Histogram (book title vs filename)")
    for bin_label, count in sorted(report.get("fuzzy_histogram", {}).items()):
        lines.append(f"- `{bin_label}`: {count}")
    lines.append("")

    if report.get("fuzzy_details_sample"):
        lines.append("## Fuzzy Match Details (sample)")
        for item in report["fuzzy_details_sample"][:10]:
            lines.append(f"- `{item['book_title']}` -> `{item['target_file']}` ratio={item['ratio']}")
        lines.append("")

    lines.append("## Family Confusion")
    for expected_fam, confused in sorted(report.get("family_confusion", {}).items()):
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(confused.items(), key=lambda x: -x[1]))
        lines.append(f"- `{expected_fam}` confused with: {pairs}")
    lines.append("")

    lines.append(f"## Rerank Drop Top20: {report.get('rerank_drop_count', 0)} samples")
    for item in report.get("rerank_drop_top20", [])[:20]:
        lines.append(f"- `{item.get('sample_id')}`: {item.get('query')}")
    lines.append("")

    lines.append("## False-Retrieval Top10")
    for f, count in report.get("false_retrieval_top10", {}).items():
        lines.append(f"- `{f}`: {count}")
    lines.append("")

    if "anchor_hit_rate" in report:
        lines.append(f"## Anchor Hit Rate: {report['anchor_hit_rate']:.1%}")
        lines.append("")

    return "\n".join(lines) + "\n"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze RAG retrieval misses")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to miss_analysis.jsonl or results.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write analysis report (default: input directory / miss_analysis_report.md)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    rows = load_jsonl(args.input)

    # If this is a results.jsonl, filter to misses only
    miss_rows = []
    results_rows = rows
    for row in rows:
        metrics = row.get("metrics") or {}
        if not (metrics.get("hit_at_5") or metrics.get("id_context_recall_at_5")):
            miss_rows.append(row)

    print(f"Analyzing {len(miss_rows)} misses out of {len(rows)} total samples", flush=True)

    report = analyze_misses(miss_rows, results_rows=results_rows)

    report_json = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    output_dir = args.input.parent
    (output_dir / "miss_analysis_report.json").write_text(report_json, encoding="utf-8")

    report_md = _render_report(report)
    output_md = args.output or (output_dir / "miss_analysis_report.md")
    output_md.write_text(report_md, encoding="utf-8")

    print(f"Wrote report to {output_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
