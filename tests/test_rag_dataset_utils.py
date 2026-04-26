import tempfile
import unittest
from pathlib import Path

from scripts.rag_dataset_utils import (
    alignment_score,
    group_rows_by_source_file,
    load_jsonl,
    loose_text,
    record_source_file,
    write_jsonl,
)


class RagDatasetUtilsTests(unittest.TestCase):
    def test_jsonl_round_trip_preserves_unicode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            rows = [{"id": "r1", "query": "\u5982\u4f55\u914d\u7f6e\u65e0\u7ebf\u7f51\u7edc\uff1f"}]

            write_jsonl(path, rows)

            self.assertEqual(load_jsonl(path), rows)

    def test_record_source_file_prefers_gold_files(self):
        row = {"gold_files": ["manual.pdf"], "metadata": {"source_file": "other.pdf"}}

        self.assertEqual(record_source_file(row), "manual.pdf")

    def test_group_rows_by_source_file(self):
        rows = [
            {"id": "a", "gold_files": ["one.pdf"]},
            {"id": "b", "gold_files": ["one.pdf"]},
            {"id": "c", "gold_files": ["two.pdf"]},
        ]

        grouped = group_rows_by_source_file(rows)

        self.assertEqual([row["id"] for row in grouped["one.pdf"]], ["a", "b"])
        self.assertEqual([row["id"] for row in grouped["two.pdf"]], ["c"])

    def test_loose_text_removes_punctuation_and_spacing(self):
        self.assertEqual(loose_text(" H3C-LA2608, wireless gateway. "), "h3cla2608wirelessgateway")

    def test_alignment_score_rewards_page_excerpt_anchor_and_keywords(self):
        row = {
            "source_excerpt": "\u5f00\u542f\u4e91\u670d\u52a1\u5e76\u70b9\u51fb\u5e94\u7528\u6309\u94ae",
            "gold_files": ["manual.pdf"],
            "gold_pages": [8],
            "expected_anchors": ["1.4.6"],
            "expected_keywords": ["\u4e91\u670d\u52a1", "\u5e94\u7528"],
        }
        chunk = {
            "filename": "manual.pdf",
            "text": "1.4.6 \u914d\u7f6e\u4e91\u670d\u52a1\n\u5f00\u542f\u4e91\u670d\u52a1\u5e76\u70b9\u51fb\u5e94\u7528\u6309\u94ae",
            "retrieval_text": "\u914d\u7f6e\u4e91\u670d\u52a1\n\u5f00\u542f\u4e91\u670d\u52a1\u5e76\u70b9\u51fb\u5e94\u7528\u6309\u94ae",
            "page_number": 8,
            "page_start": 8,
            "page_end": 8,
            "anchor_id": "1.4.6",
            "section_title": "1.4.6 \u914d\u7f6e\u4e91\u670d\u52a1",
        }

        score = alignment_score(row, chunk)

        self.assertGreaterEqual(score.score, 0.95)
        self.assertEqual(score.method, "exact")

    def test_alignment_score_downweights_neighbor_page(self):
        row = {
            "source_excerpt": "保存配置",
            "gold_files": ["manual.pdf"],
            "gold_pages": [8],
        }
        same_file_neighbor_page = {
            "filename": "manual.pdf",
            "text": "保存配置",
            "page_number": 9,
        }

        score = alignment_score(row, same_file_neighbor_page)

        self.assertIn("neighbor_page", score.reasons)
        self.assertLess(score.score, 0.75)

    def test_alignment_score_rewards_same_section_heading(self):
        row = {
            "source_excerpt": "missing excerpt",
            "gold_files": ["manual.pdf"],
            "gold_pages": [8],
            "source_heading": "1.2 配置LA2608与无线控制器互通",
        }
        chunk = {
            "filename": "manual.pdf",
            "page_number": 8,
            "section_title": "1.2 配置LA2608与无线控制器互通",
            "text": "section body",
        }

        score = alignment_score(row, chunk)

        self.assertIn("same_section", score.reasons)
        self.assertGreaterEqual(score.score, 0.45)


if __name__ == "__main__":
    unittest.main()
