from __future__ import annotations

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from answer_eval import citation_coverage, format_context_for_answer  # noqa: E402


def test_citation_coverage_matches_filename_stem_and_page_marker():
    result = citation_coverage(
        "结论来自 H3C Manual [H3C Manual.pdf p.5]。",
        expected_files=["H3C Manual.pdf"],
        expected_pages=[5],
    )

    assert result["citation_coverage"] == 1.0
    assert result["matched_files"] == ["H3C Manual.pdf"]
    assert result["matched_pages"] == ["5"]


def test_citation_coverage_partial_when_page_missing():
    result = citation_coverage(
        "结论来自 H3C Manual。",
        expected_files=["H3C Manual.pdf"],
        expected_pages=[5],
    )

    assert result["citation_coverage"] == 0.5
    assert result["matched_file_count"] == 1
    assert result["matched_page_count"] == 0


def test_format_context_for_answer_includes_metadata_and_truncates_body():
    context = format_context_for_answer(
        [
            {
                "filename": "manual.pdf",
                "page_start": 3,
                "section_path": "安装 > 配置",
                "retrieval_text": "x" * 1200,
            }
        ],
        max_chars_per_doc=20,
    )

    assert "file=manual.pdf" in context
    assert "page=3" in context
    assert "section=安装 > 配置" in context
    assert "...[truncated]" in context
