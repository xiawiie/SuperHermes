from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable


QRELS_NA = "n/a (qrels missing)"

PRIMARY_METRICS = [
    "file_hit_at_5",
    "file_page_hit_at_5",
    "file_candidate_recall_before_rerank",
    "hard_negative_context_ratio_at_5",
    "anchor_hit_at_5",
    "mrr",
    "p50_latency_ms",
    "p95_latency_ms",
]


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


def _average(rows: list[dict], metric: str) -> float | None:
    values = [
        row.get("metrics", {}).get(metric)
        for row in rows
        if isinstance(row.get("metrics", {}).get(metric), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _percentile_metric(rows: list[dict], metric: str, percentile: float) -> float | None:
    values = sorted(
        float(row.get("metrics", {}).get(metric))
        for row in rows
        if isinstance(row.get("metrics", {}).get(metric), (int, float))
    )
    return _percentile_values(values, percentile)


def _average_field(rows: list[dict], field: str) -> float | None:
    values = [row.get(field) for row in rows if isinstance(row.get(field), (int, float, bool))]
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def _percentile_field(rows: list[dict], field: str, percentile: float) -> float | None:
    values = sorted(float(row.get(field)) for row in rows if isinstance(row.get(field), (int, float, bool)))
    return _percentile_values(values, percentile)


def _percentile_values(values: list[float], percentile: float) -> float | None:
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


def _trace_distribution(rows: list[dict], key: str) -> dict[str, int]:
    return dict(
        Counter(
            str((row.get("trace") or {}).get(key))
            for row in rows
            if (row.get("trace") or {}).get(key)
        )
    )


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


def summarize_results(
    rows: list[dict],
    variants: list[str],
    *,
    pair_definitions: Iterable[tuple[str, str, str]] = (),
) -> dict[str, Any]:
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
        file_qrel_coverage = _average(variant_rows, "file_qrel_available")
        page_qrel_coverage = _average(variant_rows, "page_qrel_available")
        chunk_qrel_coverage = _average(variant_rows, "chunk_qrel_available")
        root_qrel_coverage = _average(variant_rows, "root_qrel_available")
        chunk_qrel_rows = sum(1 for row in variant_rows if row.get("metrics", {}).get("chunk_qrel_available"))
        root_qrel_rows = sum(1 for row in variant_rows if row.get("metrics", {}).get("root_qrel_available"))
        chunk_hit_at_5 = _average(variant_rows, "chunk_hit_at_5")
        root_hit_at_5 = _average(variant_rows, "root_hit_at_5")
        chunk_mrr = _average(variant_rows, "chunk_mrr")
        root_mrr = _average(variant_rows, "root_mrr")
        if chunk_qrel_coverage == 0.0:
            chunk_hit_at_5 = QRELS_NA
            chunk_mrr = QRELS_NA
        if root_qrel_coverage == 0.0:
            root_hit_at_5 = QRELS_NA
            root_mrr = QRELS_NA

        summary["variants"][variant] = {
            "rows": len(variant_rows),
            "file_qrel_coverage": file_qrel_coverage,
            "page_qrel_coverage": page_qrel_coverage,
            "chunk_qrel_coverage": chunk_qrel_coverage,
            "root_qrel_coverage": root_qrel_coverage,
            "chunk_qrel_rows": chunk_qrel_rows,
            "root_qrel_rows": root_qrel_rows,
            "hit_at_5": _average(variant_rows, "hit_at_5"),
            "initial_retrieval_hit_at_5": _average(variant_rows, "initial_retrieval_hit_at_5"),
            "final_retrieval_hit_at_5": _average(variant_rows, "final_retrieval_hit_at_5"),
            "file_hit_at_5": _average(variant_rows, "file_hit_at_5"),
            "file_page_hit_at_5": _average(variant_rows, "file_page_hit_at_5"),
            "page_hit_at_5": _average(variant_rows, "page_hit_at_5"),
            "chunk_hit_at_5": chunk_hit_at_5,
            "root_hit_at_5": root_hit_at_5,
            "chunk_mrr": chunk_mrr,
            "root_mrr": root_mrr,
            "answer_support_hit_at_5_experimental": _average(
                variant_rows,
                "answer_support_hit_at_5_experimental",
            ),
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
            "weak_any_candidate_recall_before_rerank": _average(variant_rows, "candidate_recall_before_rerank"),
            "candidate_recall_before_rerank": _average(variant_rows, "candidate_recall_before_rerank"),
            "file_candidate_recall_before_rerank": _average(variant_rows, "file_candidate_recall_before_rerank"),
            "rerank_drop_rate": _average(variant_rows, "rerank_drop_rate"),
            "file_rerank_drop_rate": _average(variant_rows, "file_rerank_drop_rate"),
            "structure_drop_rate": _average(variant_rows, "structure_drop_rate"),
            "file_structure_drop_rate": _average(variant_rows, "file_structure_drop_rate"),
            "file_rank_before_rerank_p50": _percentile_metric(variant_rows, "file_rank_before_rerank", 0.50),
            "file_rank_before_rerank_p95": _percentile_metric(variant_rows, "file_rank_before_rerank", 0.95),
            "recall_at_5": _average(variant_rows, "recall_at_5"),
            "rerank_enabled_rate": _average(variant_rows, "rerank_enabled"),
            "rerank_applied_rate": _average(variant_rows, "rerank_applied"),
            "ce_predict_executed_rate": _average(variant_rows, "ce_predict_executed"),
            "ce_cache_hit_rate": _average(variant_rows, "ce_cache_hit"),
            "avg_ce_input_count": _average(variant_rows, "ce_input_count"),
            "p50_ce_latency_ms": _percentile_metric(variant_rows, "ce_latency_ms", 0.50),
            "p95_ce_latency_ms": _percentile_metric(variant_rows, "ce_latency_ms", 0.95),
            "candidate_strategy_requested_distribution": _trace_distribution(
                variant_rows,
                "candidate_strategy_requested",
            ),
            "candidate_strategy_effective_distribution": _trace_distribution(
                variant_rows,
                "candidate_strategy_effective",
            ),
            "rerank_execution_mode_distribution": _trace_distribution(
                variant_rows,
                "rerank_execution_mode",
            ),
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
            "faithfulness_score": _average(variant_rows, "faithfulness_score"),
            "answer_relevance_score": _average(variant_rows, "answer_relevance_score"),
            "citation_coverage": _average(variant_rows, "citation_coverage"),
            "answer_eval_error_rate": _average(variant_rows, "answer_eval_error_rate"),
            "rewrite_strategy_distribution": dict(rewrite_counts),
        }
        summary["diagnostics"][variant] = dict(diagnostic_counts)

    for pair_name, old_variant, new_variant in pair_definitions:
        if old_variant in variants and new_variant in variants:
            summary["paired_comparisons"][pair_name] = _build_pairwise(rows, old_variant, new_variant)

    return summary
