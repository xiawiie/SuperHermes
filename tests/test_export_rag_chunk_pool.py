import tempfile
import unittest
from pathlib import Path

from scripts.export_rag_chunk_pool import (
    export_from_documents,
    export_from_milvus,
    normalize_chunk,
    write_chunk_pool,
)
from scripts.rag_dataset_utils import load_jsonl


class FakeMilvusManager:
    def query_all(self, filter_expr="", output_fields=None):
        self.filter_expr = filter_expr
        self.output_fields = output_fields
        return [
            {
                "chunk_id": "leaf-1",
                "root_chunk_id": "root-1",
                "filename": "manual.pdf",
                "page_number": "2",
                "text": "body",
                "retrieval_text": "heading\nbody",
                "section_title": "heading",
            }
        ]


class FakeLoader:
    def load_document(self, file_path, filename):
        return [
            {"chunk_id": "root-1", "chunk_level": 1, "filename": filename, "text": "root"},
            {"chunk_id": "leaf-1", "chunk_level": 3, "filename": filename, "text": "leaf"},
        ]


class ExportRagChunkPoolTests(unittest.TestCase):
    def test_normalize_chunk_keeps_expected_fields_and_ints(self):
        chunk = normalize_chunk({"chunk_id": "c1", "page_number": "3", "chunk_level": "3", "filename": "m.pdf"})

        self.assertEqual(chunk["chunk_id"], "c1")
        self.assertEqual(chunk["page_number"], 3)
        self.assertEqual(chunk["chunk_level"], 3)
        self.assertEqual(chunk["filename"], "m.pdf")

    def test_export_from_milvus_uses_query_all(self):
        manager = FakeMilvusManager()

        rows = export_from_milvus(manager=manager, filter_expr='filename == "manual.pdf"')

        self.assertEqual(rows[0]["chunk_id"], "leaf-1")
        self.assertIn("chunk_id", manager.output_fields)
        self.assertEqual(manager.filter_expr, 'filename == "manual.pdf"')

    def test_export_from_documents_keeps_leaf_chunks_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "manual.pdf"
            doc.write_text("placeholder", encoding="utf-8")

            rows = export_from_documents(Path(tmp), loader=FakeLoader())

        self.assertEqual([row["chunk_id"] for row in rows], ["leaf-1"])

    def test_write_chunk_pool_round_trips_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"

            write_chunk_pool(path, [{"chunk_id": "c1", "filename": "m.pdf"}])

            self.assertEqual(load_jsonl(path)[0]["chunk_id"], "c1")


if __name__ == "__main__":
    unittest.main()
