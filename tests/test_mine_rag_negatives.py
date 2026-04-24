import unittest

from scripts.mine_rag_negatives import build_contrastive_row


class MineRagNegativesTests(unittest.TestCase):
    def test_build_contrastive_row_uses_positive_and_hard_negative_files(self):
        row = {
            "id": "r1",
            "split": "train",
            "query": "\u5982\u4f55\u914d\u7f6e\u4e91\u670d\u52a1\uff1f",
            "gold_chunk_ids": ["pos-1"],
            "expected_root_ids": ["root-pos"],
            "hard_negative_files": ["other.pdf"],
            "supporting_chunks": [
                {"chunk_id": "pos-1", "root_chunk_id": "root-pos", "file_name": "manual.pdf", "text": "positive"}
            ],
        }
        chunks = [
            {"chunk_id": "pos-1", "root_chunk_id": "root-pos", "filename": "manual.pdf", "text": "positive"},
            {
                "chunk_id": "hard-1",
                "root_chunk_id": "root-hard",
                "filename": "other.pdf",
                "page_number": 3,
                "text": "\u4e91\u670d\u52a1 \u76f8\u4f3c\u4f46\u9519\u8bef",
            },
            {"chunk_id": "easy-1", "root_chunk_id": "root-easy", "filename": "easy.pdf", "page_number": 1, "text": "unrelated"},
        ]

        out = build_contrastive_row(row, chunks, min_hard=1, min_easy=1)

        self.assertEqual(out["positive_contexts"][0]["chunk_id"], "pos-1")
        self.assertEqual(out["hard_negatives"][0]["chunk_id"], "hard-1")
        self.assertEqual(out["easy_negatives"][0]["chunk_id"], "easy-1")

    def test_does_not_use_positive_as_negative(self):
        row = {
            "id": "r1",
            "split": "train",
            "query": "q",
            "gold_chunk_ids": ["pos-1"],
            "expected_root_ids": ["root-pos"],
            "supporting_chunks": [{"chunk_id": "pos-1", "root_chunk_id": "root-pos", "file_name": "manual.pdf"}],
        }
        chunks = [{"chunk_id": "pos-1", "root_chunk_id": "root-pos", "filename": "manual.pdf", "text": "positive"}]

        out = build_contrastive_row(row, chunks, min_hard=1, min_easy=1)

        self.assertEqual(out["hard_negatives"], [])
        self.assertEqual(out["easy_negatives"], [])


if __name__ == "__main__":
    unittest.main()
