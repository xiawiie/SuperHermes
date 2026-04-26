from __future__ import annotations

from typing import Any


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def dataset_schema_versions(records: list[dict]) -> list[str]:
    versions = {
        str(record.get("benchmark_schema_version") or "")
        for record in records
        if record.get("benchmark_schema_version")
    }
    return sorted(versions)


def validate_eval_dataset_records(records: list[dict], dataset: str) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    required_any = {
        "query": ("query", "question", "input"),
        "reference_answer": ("reference_answer", "expected_answer"),
        "expected_files": ("expected_files", "gold_files"),
        "expected_pages": ("expected_pages", "expected_page_refs", "gold_pages", "gold_doc_ids"),
    }
    required_exact = ("positive_contexts", "relevance_judgments", "hard_negative_files", "expected_keyword_policy")

    for idx, record in enumerate(records, 1):
        row_id = record.get("sample_id") or record.get("id") or idx
        for label, candidates in required_any.items():
            if not any(record.get(name) for name in candidates):
                errors.append({"row": idx, "id": row_id, "error": f"missing_{label}"})
        for field in required_exact:
            if not record.get(field):
                errors.append({"row": idx, "id": row_id, "error": f"missing_{field}"})

        policy = record.get("expected_keyword_policy")
        if policy and not isinstance(policy, dict):
            errors.append({"row": idx, "id": row_id, "error": "expected_keyword_policy_not_object"})
        elif isinstance(policy, dict):
            try:
                min_match = int(policy.get("min_match", 0))
                total = int(policy.get("total", 0))
            except (TypeError, ValueError):
                errors.append({"row": idx, "id": row_id, "error": "expected_keyword_policy_invalid_numbers"})
            else:
                if min_match < 1 or (total and min_match > total):
                    errors.append({"row": idx, "id": row_id, "error": "expected_keyword_policy_invalid_range"})

    return {
        "ok": not errors,
        "dataset": dataset,
        "row_count": len(records),
        "schema_versions": dataset_schema_versions(records),
        "errors": errors[:50],
        "error_count": len(errors),
    }
