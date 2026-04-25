"""Tests for filename normalization and matching."""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from query_plan import _normalize_filename, _filename_match_score


class TestNormalizeFilename:
    def test_strips_pdf_extension(self):
        assert _normalize_filename("test.pdf") == "test"

    def test_strips_docx_extension(self):
        assert _normalize_filename("doc.docx") == "doc"

    def test_strips_xlsx_extension(self):
        assert _normalize_filename("sheet.xlsx") == "sheet"

    def test_removes_copy_suffix(self):
        assert _normalize_filename("file_副本.pdf") == "file"

    def test_removes_parenthetical_number(self):
        assert _normalize_filename("file(1).pdf") == "file"

    def test_removes_chinese_parenthetical(self):
        assert _normalize_filename("file（副本）.pdf") == "file"

    def test_lowercases(self):
        assert _normalize_filename("MyFile.PDF") == "myfile"

    def test_collapses_whitespace(self):
        assert _normalize_filename("hello   world.pdf") == "hello world"


class TestFilenameMatchScore:
    def test_exact_match(self):
        assert _filename_match_score("test", "test") == 1.0

    def test_contains_match(self):
        assert _filename_match_score("test", "test file") == 0.95

    def test_reverse_contains(self):
        assert _filename_match_score("test file", "test") == 0.95

    def test_no_match(self):
        score = _filename_match_score("xyz", "abc")
        assert 0.0 <= score < 0.5

    def test_partial_token_coverage(self):
        score = _filename_match_score("h3c la2608", "h3c la2608室内无线网关")
        assert score >= 0.5

    def test_chinese_filename_match(self):
        score = _filename_match_score(
            "h3c la2608室内无线网关",
            "h3c la2608室内无线网关 用户手册-6w100-整本手册",
        )
        assert score >= 0.8

    def test_model_number_match(self):
        score = _filename_match_score("la2608", "h3c la2608室内无线网关 用户手册")
        assert score >= 0.3
