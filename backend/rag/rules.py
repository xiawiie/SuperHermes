"""Shared anchor/title normalization rules for RAG ingestion and retrieval."""
from __future__ import annotations

import re


ANCHOR_PATTERN = re.compile(
    r"(第[一二三四五六七八九十百千万零两0-9]+[编章节条部分款项]|"
    r"\d+(?:\.\d+){1,4}|"
    r"[一二三四五六七八九十]+、|"
    r"[（(][一二三四五六七八九十0-9A-Za-z]+[)）]|"
    r"附录[A-Za-z0-9一二三四五六七八九十]+|"
    r"附件[0-9一二三四五六七八九十]+)"
)

ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千万零两0-9]+[编章节条部分款项]\s*")
DECIMAL_PATTERN = re.compile(r"^(\d+(?:\.\d+){0,4})\s+\S+")
LIST_PATTERN = re.compile(r"^[一二三四五六七八九十]+、\s*\S+")
PAREN_LIST_PATTERN = re.compile(r"^[（(][一二三四五六七八九十0-9A-Za-z]+[)）]\s*\S+")

TITLE_BLACKLIST = frozenset({
    "目录",
    "前言",
    "附录",
    "附录a",
    "附录b",
    "附件",
    "封面",
    "修订记录",
    "目录页",
})


def extract_anchor_id(title: str | None) -> str:
    if not title:
        return ""
    match = ANCHOR_PATTERN.search(title.strip())
    return match.group(0) if match else ""


def normalize_title(value: str | None) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    return text.strip(" -:：|")
