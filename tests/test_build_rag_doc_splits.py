import unittest

from scripts.build_rag_doc_splits import assign_file_splits, derive_frozen_eval_rows


class BuildRagDocSplitsTests(unittest.TestCase):
    def test_assign_file_splits_keeps_files_exclusive(self):
        rows = []
        for idx in range(10):
            rows.append({"id": f"r{idx}", "gold_files": [f"manual-{idx}.pdf"], "answer_type": "operation"})

        manifest = assign_file_splits(rows, dev_files=2, test_files=3)

        train = set(manifest["splits"]["train"])
        dev = set(manifest["splits"]["dev"])
        test = set(manifest["splits"]["test"])
        self.assertFalse(train & dev)
        self.assertFalse(train & test)
        self.assertFalse(dev & test)
        self.assertEqual(len(dev), 2)
        self.assertEqual(len(test), 3)
        self.assertEqual(len(train | dev | test), 10)

    def test_derive_frozen_eval_rows_excludes_train(self):
        rows = [
            {"id": "train-row", "gold_files": ["train.pdf"]},
            {"id": "dev-row", "gold_files": ["dev.pdf"]},
            {"id": "test-row", "gold_files": ["test.pdf"]},
        ]
        manifest = {
            "splits": {
                "train": ["train.pdf"],
                "dev": ["dev.pdf"],
                "test": ["test.pdf"],
            }
        }

        frozen = derive_frozen_eval_rows(rows, manifest)

        self.assertEqual([row["id"] for row in frozen], ["dev-row", "test-row"])
        self.assertEqual([row["split"] for row in frozen], ["dev", "test"])


if __name__ == "__main__":
    unittest.main()
