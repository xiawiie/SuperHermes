import unittest

from scripts.validate_rag_dataset import (
    validate_chunk_gold_rows,
    validate_contrastive_rows,
    validate_split_manifest,
)


class ValidateRagDatasetTests(unittest.TestCase):
    def test_validate_split_manifest_detects_leakage(self):
        manifest = {"splits": {"train": ["a.pdf"], "dev": ["a.pdf"], "test": []}}

        report = validate_split_manifest(manifest)

        self.assertFalse(report["ok"])
        self.assertIn("split_file_overlap", report["errors"])

    def test_validate_chunk_gold_requires_supporting_chunks_for_aligned_rows(self):
        rows = [{"id": "r1", "quality": {"alignment_status": "aligned"}, "gold_chunk_ids": ["c1"], "supporting_chunks": []}]

        report = validate_chunk_gold_rows(rows)

        self.assertFalse(report["ok"])
        self.assertIn("aligned_row_without_supporting_chunks", report["errors"])

    def test_validate_contrastive_detects_positive_negative_collision(self):
        rows = [
            {
                "id": "r1",
                "positive_contexts": [{"chunk_id": "c1"}],
                "hard_negatives": [{"chunk_id": "c1"}],
                "easy_negatives": [],
            }
        ]

        report = validate_contrastive_rows(rows)

        self.assertFalse(report["ok"])
        self.assertIn("positive_negative_collision", report["errors"])


if __name__ == "__main__":
    unittest.main()
