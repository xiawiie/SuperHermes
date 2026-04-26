from __future__ import annotations

from typing import Any


CORE_SUMMARY_FIELDS = (
    "file_qrel_coverage",
    "page_qrel_coverage",
    "chunk_qrel_coverage",
    "root_qrel_coverage",
    "file_hit_at_5",
    "file_page_hit_at_5",
    "chunk_hit_at_5",
    "root_hit_at_5",
    "mrr",
    "file_candidate_recall_before_rerank",
    "hard_negative_context_ratio_at_5",
)


def extract_core_summary(summary: dict[str, Any]) -> dict[str, Any]:
    variants = summary.get("variants") or {}
    return {
        variant: {field: metrics.get(field) for field in CORE_SUMMARY_FIELDS}
        for variant, metrics in sorted(variants.items())
    }


def compare_core_summary(
    old_summary: dict[str, Any],
    new_summary: dict[str, Any],
    *,
    float_tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    old_core = extract_core_summary(old_summary)
    new_core = extract_core_summary(new_summary)
    diffs: list[dict[str, Any]] = []
    for variant in sorted(set(old_core) | set(new_core)):
        old_metrics = old_core.get(variant, {})
        new_metrics = new_core.get(variant, {})
        for field in sorted(set(old_metrics) | set(new_metrics)):
            old_value = old_metrics.get(field)
            new_value = new_metrics.get(field)
            if isinstance(old_value, float) and isinstance(new_value, float):
                if abs(old_value - new_value) <= float_tolerance:
                    continue
            elif old_value == new_value:
                continue
            diffs.append(
                {
                    "variant": variant,
                    "field": field,
                    "old": old_value,
                    "new": new_value,
                }
            )
    return diffs
