"""Tests for filename normalization via query_plan._normalize_filename."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from query_plan import _normalize_filename


class TestFilenameNormalization:
    def test_pdf(self):
        assert _normalize_filename("manual.pdf") == "manual"

    def test_docx(self):
        assert _normalize_filename("guide.docx") == "guide"

    def test_xlsx(self):
        assert _normalize_filename("data.xlsx") == "data"

    def test_no_extension(self):
        assert _normalize_filename("readme") == "readme"

    def test_copy_suffix(self):
        assert _normalize_filename("file_副本.pdf") == "file"

    def test_paren_number(self):
        assert _normalize_filename("doc(1).pdf") == "doc"

    def test_chinese_paren(self):
        assert _normalize_filename("doc（副本）.pdf") == "doc"

    def test_lowercases(self):
        assert _normalize_filename("MyFile.PDF") == "myfile"

    def test_collapse_spaces(self):
        assert _normalize_filename("hello   world.pdf") == "hello world"

    def test_complex_filename(self):
        result = _normalize_filename("H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf")
        assert "h3c" in result
        assert "la2608" in result

    def test_empty_string(self):
        assert _normalize_filename("") == ""
