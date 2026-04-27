"""QueryPlan: parse user query into semantic_query + metadata constraints + route.

Design principles (from all-in-rag):
  - Query Construction: separate semantic_query from metadata_filter
  - Query Routing: rule-based routing to scoped_hybrid or global_hybrid
  - Model numbers are only removed from semantic_query when high-confidence
    file matches occur (scope_mode in {filter, boost}); otherwise retained.
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal

import jieba

from backend.config import MILVUS_COLLECTION
from backend.shared.filename_normalization import normalize_filename_for_match

logger = logging.getLogger(__name__)

# --- Configuration from env ---
DOC_SCOPE_MATCH_FILTER = float(os.getenv("DOC_SCOPE_MATCH_FILTER", "0.85"))
DOC_SCOPE_MATCH_BOOST = float(os.getenv("DOC_SCOPE_MATCH_BOOST", "0.60"))
DOC_SCOPE_FILENAME_REGISTRY_REFRESH_SECONDS = int(
    os.getenv("DOC_SCOPE_FILENAME_REGISTRY_REFRESH_SECONDS", "600")
)
DOC_SCOPE_MATCH_TRACE_MIN = float(os.getenv("DOC_SCOPE_MATCH_TRACE_MIN", "0.30"))
_REGISTRY_CACHE_MAX_KEYS = int(os.getenv("DOC_SCOPE_FILENAME_REGISTRY_CACHE_KEYS", "8"))

# --- Regex patterns ---
_BOOK_TITLE_RE = re.compile(r"《([^》]+)》")
_MODEL_NUMBER_RE = re.compile(r"[A-Z]{2,}\d{3,}[A-Z0-9]*")
_CHAPTER_RE = re.compile(r"第\s*\d+\s*章|附录\s*[A-Z\d]")
_BOOK_TITLE_PREFIX_RE = re.compile(r"《[^》]+》\s*中[，,]?\s*")


@dataclass
class QueryPlan:
    raw_query: str
    semantic_query: str
    clean_query: str
    doc_hints: list[str] = field(default_factory=list)
    matched_files: list[tuple[str, float]] = field(default_factory=list)
    scope_mode: Literal["filter", "boost", "none"] = "none"
    heading_hint: str | None = None
    anchors: list[str] = field(default_factory=list)
    model_numbers: list[str] = field(default_factory=list)
    intent_type: str | None = None
    route: Literal["scoped_hybrid", "global_hybrid"] = "global_hybrid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "semantic_query": self.semantic_query,
            "clean_query": self.clean_query,
            "doc_hints": self.doc_hints,
            "matched_files": [(f, round(s, 3)) for f, s in self.matched_files],
            "scope_mode": self.scope_mode,
            "heading_hint": self.heading_hint,
            "anchors": self.anchors,
            "model_numbers": self.model_numbers,
            "intent_type": self.intent_type,
            "route": self.route,
        }


def _normalize_filename(name: str) -> str:
    """Normalize filename for matching: strip extension, lower, remove suffixes."""
    return normalize_filename_for_match(name)


def _filename_match_score(query_hint: str, filename_norm: str) -> float:
    """Compute compound match score between a query hint and normalized filename."""
    if query_hint == filename_norm:
        return 1.0
    if query_hint in filename_norm or filename_norm in query_hint:
        return 0.95

    hint_tokens = set(jieba.cut(query_hint))
    file_tokens = set(jieba.cut(filename_norm))
    token_coverage = len(hint_tokens & file_tokens) / max(len(hint_tokens), 1)

    seq_ratio = SequenceMatcher(None, query_hint, filename_norm).ratio()

    return max(token_coverage, seq_ratio)


def _match_doc_hints(
    doc_hints: list[str],
    filename_registry: list[dict[str, str]],
) -> list[tuple[str, float]]:
    """Match doc_hints against filename registry, return (filename, score) pairs."""
    scored: list[tuple[str, float]] = []
    for hint in doc_hints:
        hint_norm = _normalize_filename(hint)
        if not hint_norm:
            continue
        for entry in filename_registry:
            score = _filename_match_score(hint_norm, entry["normalized"])
            if score >= DOC_SCOPE_MATCH_TRACE_MIN:
                scored.append((entry["raw"], score))

    # Deduplicate by filename, keeping best score
    best: dict[str, float] = {}
    for filename, score in scored:
        best[filename] = max(best.get(filename, 0.0), score)

    result = sorted(best.items(), key=lambda x: -x[1])
    return result


# --- Lazy filename registry ---

_registry_cache: OrderedDict[str, tuple[list[dict[str, str]], float]] = OrderedDict()


def _index_version_from_cache(cache_client: Any = None) -> str:
    if not cache_client:
        return "0"
    try:
        if hasattr(cache_client, "get_string"):
            return str(cache_client.get_string("milvus_index_version") or "0")
        value = cache_client.get("milvus_index_version")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value or "0")
    except Exception:
        return "0"


def _registry_cache_key(collection: str, index_version: str) -> str:
    # RedisCache prefixes this logical key with "superhermes:", yielding
    # superhermes:filename_registry:{collection}:v{milvus_index_version}.
    return f"filename_registry:{collection}:v{index_version}"


def _remember_registry(cache_key: str, entries: list[dict[str, str]], cached_at: float) -> None:
    _registry_cache[cache_key] = (entries, cached_at)
    _registry_cache.move_to_end(cache_key)
    while len(_registry_cache) > max(1, _REGISTRY_CACHE_MAX_KEYS):
        _registry_cache.popitem(last=False)


def _registry_from_process_cache(cache_key: str, now: float, refresh_interval: int) -> list[dict[str, str]] | None:
    cached = _registry_cache.get(cache_key)
    if not cached:
        return None
    cached_entries, cached_at = cached
    if now - cached_at >= refresh_interval:
        return None
    _registry_cache.move_to_end(cache_key)
    return cached_entries


def _registry_from_redis(cache_client: Any, cache_key: str) -> list[dict[str, str]]:
    if not cache_client:
        return []
    try:
        if hasattr(cache_client, "get_json"):
            value = cache_client.get_json(cache_key)
            return value if isinstance(value, list) else []
        return _decode_registry(cache_client.get(cache_key))
    except Exception:
        return []


def _store_registry_in_redis(
    cache_client: Any,
    cache_key: str,
    entries: list[dict[str, str]],
    ttl_seconds: int,
) -> None:
    if not cache_client or not entries:
        return
    try:
        if hasattr(cache_client, "set_json"):
            cache_client.set_json(cache_key, entries, ttl=ttl_seconds)
            return
        import json

        cache_client.setex(cache_key, ttl_seconds, json.dumps(entries, ensure_ascii=False))
    except Exception:
        return


def get_filename_registry(milvus_manager: Any, cache_client: Any = None) -> list[dict[str, str]]:
    """Get or refresh the filename registry from Milvus (with Redis caching)."""
    collection = MILVUS_COLLECTION
    index_version = _index_version_from_cache(cache_client)
    cache_key = _registry_cache_key(collection, index_version)
    now = time.time()
    refresh_interval = DOC_SCOPE_FILENAME_REGISTRY_REFRESH_SECONDS

    cached_entries = _registry_from_process_cache(cache_key, now, refresh_interval)
    if cached_entries is not None:
        return cached_entries

    entries = _registry_from_redis(cache_client, cache_key)
    if entries:
        _remember_registry(cache_key, entries, now)
        return entries

    # Query Milvus for distinct filenames
    entries = _query_filenames_from_milvus(milvus_manager)

    _store_registry_in_redis(cache_client, cache_key, entries, refresh_interval * 2)
    _remember_registry(cache_key, entries, now)
    return entries


def _decode_registry(data: Any) -> list[dict[str, str]]:
    """Decode registry data from Redis cache."""
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    if isinstance(data, str):
        import json
        try:
            return json.loads(data)
        except Exception:
            return []
    return []


def _query_filenames_from_milvus(milvus_manager: Any) -> list[dict[str, str]]:
    """Query Milvus for distinct filenames at leaf level."""
    entries: list[dict[str, str]] = []
    try:
        leaf_level = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))
        filenames = milvus_manager.query_unique_filenames(
            filter_expr=f"chunk_level == {leaf_level}",
        )
        for f in filenames:
            entries.append({
                "raw": f,
                "normalized": _normalize_filename(f),
            })
    except Exception as exc:
        logger.warning("Failed to query filename registry from Milvus: %s", exc)
    return entries


# --- Main parser ---

def parse_query_plan(
    raw_query: str,
    filename_registry: list[dict[str, str]] | None = None,
    context_files: list[str] | None = None,
) -> QueryPlan:
    """Parse raw query into a QueryPlan with semantic_query, doc_hints, and route.

    Args:
        raw_query: Original user query.
        filename_registry: List of {"raw": ..., "normalized": ...} dicts.
        context_files: User-explicit context files (highest priority).
    """
    # 1. Extract 《》 book titles
    book_titles = _BOOK_TITLE_RE.findall(raw_query)
    doc_hints = list(book_titles)

    # 2. Extract model numbers
    model_numbers = _MODEL_NUMBER_RE.findall(raw_query)

    # 3. Extract chapter/appendix anchors
    anchors = _CHAPTER_RE.findall(raw_query)

    # 4. Extract heading hint (text after book title prefix)
    heading_hint = None
    clean_query = _BOOK_TITLE_PREFIX_RE.sub("", raw_query).strip()
    if not clean_query:
        clean_query = raw_query

    # Try to extract heading from the cleaned query
    heading_match = re.match(r"^(?:如何|怎么|怎样)?\s*(.+?)[？?]?\s*$", clean_query)
    if heading_match:
        heading_hint = heading_match.group(1).strip()

    # 5. Match doc_hints against filename registry
    matched_files: list[tuple[str, float]] = []
    scope_mode: Literal["filter", "boost", "none"] = "none"

    if filename_registry and doc_hints:
        matched_files = _match_doc_hints(doc_hints, filename_registry)

        # Also try matching model numbers against filenames
        if model_numbers:
            model_hints = matched_files[:]
            for mn in model_numbers:
                mn_matches = _match_doc_hints([mn], filename_registry)
                for f, s in mn_matches:
                    # Only add if not already present with higher score
                    existing = [i for i, (ef, _) in enumerate(model_hints) if ef == f]
                    if existing:
                        idx = existing[0]
                        model_hints[idx] = (f, max(model_hints[idx][1], s))
                    else:
                        model_hints.append((f, s))
            matched_files = sorted(model_hints, key=lambda x: -x[1])

    # Determine scope_mode based on best match score
    routable_matches = [(f, score) for f, score in matched_files if score >= DOC_SCOPE_MATCH_BOOST]
    if routable_matches:
        best_score = routable_matches[0][1]
        if best_score >= DOC_SCOPE_MATCH_FILTER:
            scope_mode = "filter"
        elif best_score >= DOC_SCOPE_MATCH_BOOST:
            scope_mode = "boost"

    # 6. Build semantic_query
    semantic_query = _BOOK_TITLE_PREFIX_RE.sub("", raw_query).strip()

    # Conditionally remove model numbers from semantic_query
    # Only when scope_mode in {filter, boost} (high-confidence file match)
    if scope_mode in {"filter", "boost"} and model_numbers:
        for mn in model_numbers:
            semantic_query = semantic_query.replace(mn, "").strip()
        semantic_query = re.sub(r"\s+", " ", semantic_query).strip()

    if not semantic_query:
        semantic_query = raw_query

    # 7. User context_files override: highest priority
    if context_files:
        # User explicitly selected files -> force scoped mode
        scope_mode = "filter"
        matched_files = [(f, 1.0) for f in context_files]

    # 8. Route determination (tightened rules)
    route: Literal["scoped_hybrid", "global_hybrid"]
    if scope_mode in {"filter", "boost"}:
        route = "scoped_hybrid"
    else:
        route = "global_hybrid"

    return QueryPlan(
        raw_query=raw_query,
        semantic_query=semantic_query,
        clean_query=clean_query,
        doc_hints=doc_hints,
        matched_files=matched_files,
        scope_mode=scope_mode,
        heading_hint=heading_hint,
        anchors=anchors,
        model_numbers=model_numbers,
        intent_type=None,
        route=route,
    )
