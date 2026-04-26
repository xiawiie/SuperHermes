"""RAG index profile helpers.

Profiles let evaluation/index variants keep separate Milvus collections,
BM25 state files, parent chunks, and traces without changing user-facing
chunk ids. The legacy profile intentionally keeps unprefixed storage keys for
backward compatibility with existing indexes.
"""
from __future__ import annotations

import os
import re

LEGACY_INDEX_PROFILE = "legacy"
_SAFE_PROFILE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def normalize_index_profile(value: str | None) -> str:
    profile = (value or "").strip()
    if not profile:
        return LEGACY_INDEX_PROFILE
    profile = _SAFE_PROFILE_RE.sub("_", profile)
    return profile[:120] or LEGACY_INDEX_PROFILE


def current_index_profile() -> str:
    return normalize_index_profile(os.getenv("RAG_INDEX_PROFILE"))


def storage_chunk_id(chunk_id: str, profile: str | None = None) -> str:
    clean_id = (chunk_id or "").strip()
    clean_profile = normalize_index_profile(profile or current_index_profile())
    if not clean_id or clean_profile == LEGACY_INDEX_PROFILE:
        return clean_id
    return f"{clean_profile}::{clean_id}"


def display_chunk_id(chunk_id: str, profile: str | None = None) -> str:
    clean_id = (chunk_id or "").strip()
    clean_profile = normalize_index_profile(profile or current_index_profile())
    prefix = f"{clean_profile}::"
    if clean_profile != LEGACY_INDEX_PROFILE and clean_id.startswith(prefix):
        return clean_id[len(prefix):]
    return clean_id
