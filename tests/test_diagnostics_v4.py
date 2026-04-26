"""Tests for v4 diagnostics five-category classification."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag.diagnostics import classify_failure
from scripts.analyze_rag_misses import analyze_misses


class TestFiveCategoryDiagnostics:
    def test_file_recall_miss(self):
        """Correct file not in candidates -> file_recall_miss."""
        result = classify_failure(
            query="如何配置LA2608？",
            rag_trace={
                "retrieved_chunks": [
                    {"filename": "other_device.pdf", "score": 0.8},
                ],
                "candidates_before_rerank": [
                    {"filename": "other_device.pdf"},
                ],
            },
            expected_files=["H3C LA2608室内无线网关 用户手册.pdf"],
        )
        assert result["category"] == "file_recall_miss"

    def test_hard_negative_confusion(self):
        """All top5 from hard negatives -> hard_negative_confusion."""
        result = classify_failure(
            query="如何配置LA2608？",
            rag_trace={
                "retrieved_chunks": [
                    {"filename": "wrong1.pdf", "score": 0.9},
                    {"filename": "wrong2.pdf", "score": 0.8},
                ],
            },
            expected_files=["correct.pdf"],
            hard_negative_files=["wrong1.pdf", "wrong2.pdf"],
        )
        assert result["category"] == "hard_negative_confusion"

    def test_ranking_miss(self):
        """File in candidates but not in top5 -> ranking_miss."""
        result = classify_failure(
            query="如何配置LA2608？",
            rag_trace={
                "retrieved_chunks": [
                    {"filename": "other_device.pdf", "score": 0.9},
                ],
                "candidates_before_rerank": [
                    {"filename": "correct.pdf", "score": 0.5},
                    {"filename": "other_device.pdf", "score": 0.8},
                ],
            },
            expected_files=["correct.pdf"],
        )
        assert result["category"] == "ranking_miss"

    def test_low_confidence(self):
        """Top1 score below threshold -> low_confidence."""
        result = classify_failure(
            query="如何配置LA2608？",
            rag_trace={
                "retrieved_chunks": [
                    {"filename": "some.pdf", "score": 0.05},
                ],
            },
        )
        assert result["category"] == "low_confidence"

    def test_insufficient_trace(self):
        """Missing rag_trace -> insufficient_trace."""
        result = classify_failure(
            query="test",
            rag_trace=None,
        )
        assert result["category"] == "insufficient_trace"

    def test_ok_when_matched(self):
        """Matched with file and keywords and page -> ok."""
        result = classify_failure(
            query="test",
            rag_trace={
                "retrieved_chunks": [
                    {"filename": "correct.pdf", "section_title": "test section", "page_number": 5, "retrieval_text": "test section keyword test"},
                ],
                "candidates_before_rerank": [
                    {"filename": "correct.pdf", "section_title": "test section", "retrieval_text": "test section keyword test"},
                ],
            },
            expected_files=["correct.pdf"],
            expected_keywords=["test"],
        )
        # Without expected_pages, it should match on file + keyword
        assert result["category"] in {"ok", "ranking_miss"}

    def test_file_only_hit_is_ok(self):
        result = classify_failure(
            query="test",
            rag_trace={
                "retrieved_chunks": [{"filename": "correct.pdf", "score": 0.9}],
                "candidates_before_rerank": [{"filename": "correct.pdf", "score": 0.9}],
            },
            expected_files=["correct.pdf"],
        )

        assert result["category"] == "ok"

    def test_page_match_uses_page_start_end(self):
        result = classify_failure(
            query="test",
            rag_trace={
                "retrieved_chunks": [{"filename": "correct.pdf", "page_start": 4, "page_end": 6, "score": 0.9}],
                "candidates_before_rerank": [{"filename": "correct.pdf", "page_start": 4, "page_end": 6}],
            },
            expected_files=["correct.pdf"],
            expected_pages=[5],
        )

        assert result["category"] == "ok"


class TestMissAnalysisScript:
    def test_stage_candidates_drive_ranking_miss_classification(self):
        report = analyze_misses(
            [
                {
                    "sample_id": "s1",
                    "query": "q",
                    "expected": {"expected_files": ["correct.pdf"]},
                    "metrics": {"candidate_recall_before_rerank": 1.0},
                    "retrieved_chunks": [{"filename": "other.pdf", "score": 0.9}],
                    "stage_candidates": {
                        "before_rerank": [{"filename": "correct.pdf", "score": 0.5}],
                    },
                }
            ]
        )

        assert report["category_counts"] == {"ranking_miss": 1}

    def test_results_rows_with_file_page_hit_classify_ok(self):
        report = analyze_misses(
            [
                {
                    "sample_id": "s1",
                    "query": "q",
                    "expected": {"expected_files": ["correct.pdf"], "expected_pages": [5]},
                    "metrics": {"file_page_hit_at_5": True},
                    "retrieved_chunks": [{"filename": "correct.pdf", "page_start": 4, "page_end": 6, "score": 0.9}],
                }
            ]
        )

        assert report["category_counts"] == {"ok": 1}
