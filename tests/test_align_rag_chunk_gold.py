import unittest

from scripts.align_rag_chunk_gold import align_row, build_chunk_gold_rows


class AlignRagChunkGoldTests(unittest.TestCase):
    def test_align_row_selects_same_page_excerpt_chunk(self):
        row = {
            "id": "r1",
            "question": "\u5982\u4f55\u914d\u7f6e\u4e91\u670d\u52a1\uff1f",
            "query": "\u5982\u4f55\u914d\u7f6e\u4e91\u670d\u52a1\uff1f",
            "expected_answer": "\u5f00\u542f\u4e91\u670d\u52a1\u5e76\u70b9\u51fb\u5e94\u7528\u6309\u94ae",
            "source_excerpt": "\u5f00\u542f\u4e91\u670d\u52a1\u5e76\u70b9\u51fb\u5e94\u7528\u6309\u94ae",
            "gold_files": ["manual.pdf"],
            "gold_pages": [8],
            "gold_doc_ids": ["manual.pdf::p8"],
            "expected_anchors": ["1.4.6"],
            "expected_keywords": ["\u4e91\u670d\u52a1", "\u5e94\u7528"],
            "answer_type": "operation",
            "quality_checks": {"quality_score": 6.1},
        }
        chunks = [
            {
                "chunk_id": "wrong",
                "root_chunk_id": "root-wrong",
                "filename": "manual.pdf",
                "page_number": 9,
                "text": "\u5f00\u542f\u5176\u5b83\u670d\u52a1",
            },
            {
                "chunk_id": "leaf-1",
                "root_chunk_id": "root-1",
                "parent_chunk_id": "root-1",
                "filename": "manual.pdf",
                "page_number": 8,
                "page_start": 8,
                "page_end": 8,
                "anchor_id": "1.4.6",
                "section_title": "1.4.6 \u914d\u7f6e\u4e91\u670d\u52a1",
                "text": "\u5f00\u542f\u4e91\u670d\u52a1\u5e76\u70b9\u51fb\u5e94\u7528\u6309\u94ae",
                "retrieval_text": "\u914d\u7f6e\u4e91\u670d\u52a1\n\u5f00\u542f\u4e91\u670d\u52a1\u5e76\u70b9\u51fb\u5e94\u7528\u6309\u94ae",
            },
        ]

        aligned = align_row(row, chunks, split="train")

        self.assertEqual(aligned["gold_chunk_ids"], ["leaf-1"])
        self.assertEqual(aligned["expected_root_ids"], ["root-1"])
        self.assertTrue(aligned["canonical_chunk_ids"])
        self.assertTrue(aligned["canonical_root_ids"])
        self.assertEqual(aligned["root_type"], "section")
        self.assertEqual(aligned["quality"]["alignment_status"], "aligned")
        self.assertEqual(aligned["supporting_chunks"][0]["match_method"], "exact")

    def test_build_chunk_gold_rows_marks_failed_alignment(self):
        rows = [{"id": "r1", "gold_files": ["manual.pdf"], "gold_pages": [1], "source_excerpt": "missing"}]
        manifest = {"splits": {"train": ["manual.pdf"], "dev": [], "test": []}}

        out = build_chunk_gold_rows(rows, [], manifest)

        self.assertEqual(out[0]["quality"]["alignment_status"], "failed")
        self.assertEqual(out[0]["quality"]["alignment_failure_reason"], "no_candidate_same_file")

    def test_align_row_marks_ambiguous_low_confidence_candidates_for_review(self):
        row = {
            "id": "r1",
            "source_excerpt": "打开蓝牙并保存设置",
            "gold_files": ["manual.pdf"],
            "gold_pages": [3],
            "expected_anchors": ["蓝牙设置"],
            "expected_keywords": ["蓝牙"],
        }
        chunks = [
            {
                "chunk_id": "leaf-1",
                "root_chunk_id": "root-1",
                "filename": "manual.pdf",
                "page_number": 3,
                "section_title": "蓝牙设置",
                "text": "蓝牙设置",
            }
        ]

        aligned = align_row(row, chunks, split="dev")

        self.assertEqual(aligned["quality"]["alignment_status"], "ambiguous")
        self.assertEqual(aligned["quality"]["review_status"], "needs_review")
        self.assertEqual(aligned["gold_chunk_ids"], [])
        self.assertTrue(aligned["supporting_chunks"])


if __name__ == "__main__":
    unittest.main()
