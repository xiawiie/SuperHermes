from __future__ import annotations

import math
from typing import Any

from backend.shared.filename_normalization import normalize_filename_for_match


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    text = str(value)
    return [text] if text else []


def normalized_filename_set(values: Any) -> set[str]:
    normalized: set[str] = set()
    for item in as_list(values):
        value = normalize_filename_for_match(item)
        if value:
            normalized.add(value)
    return normalized


def doc_filename_norm(doc: dict) -> str:
    return normalize_filename_for_match(doc.get("filename"))


def rate(count: int, total: int) -> float:
    return (count / total) if total else 0.0


def percentile_values(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def present(value: Any) -> bool:
    return value is not None and str(value).strip() != ""
