from __future__ import annotations

import os
import re
import unicodedata


def raw_filename_basename(value: str | None) -> str:
    """Return a stable basename for trace output without stripping extensions."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\\", "/")
    return os.path.basename(text).strip()


def normalize_filename_for_match(value: str | None, *, strip_extension: bool = True) -> str:
    """Normalize filenames for matching while preserving raw names elsewhere."""
    base = raw_filename_basename(value)
    if strip_extension:
        base = os.path.splitext(base)[0]
    base = re.sub(r"\s+", " ", base).strip().lower()
    base = re.sub(r"[_\s]*副本$", "", base)
    base = re.sub(r"\(\d+\)$", "", base)
    base = re.sub(r"\([^)]*\)", "", base)
    base = re.sub(r"（[^）]*）", "", base)
    return base.strip()
