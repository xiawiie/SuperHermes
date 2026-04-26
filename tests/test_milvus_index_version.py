import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import backend.infra.vector_store.milvus_client as milvus_client  # noqa: E402
import backend.infra.vector_store.milvus_writer as milvus_writer  # noqa: E402


class FakeCache:
    def __init__(self):
        self.incr_keys = []

    def incr(self, key):
        self.incr_keys.append(key)
        return len(self.incr_keys)


class MilvusIndexVersionTests(unittest.TestCase):
    def test_effective_hnsw_ef_is_strictly_larger_than_limit(self):
        self.assertEqual(milvus_client._effective_hnsw_ef(64, 100), 101)
        self.assertEqual(milvus_client._effective_hnsw_ef(128, 100), 128)

    def test_writer_bumps_index_version_once_after_successful_insert(self):
        class FakeEmbeddingService:
            def get_all_embeddings(self, texts):
                return [[0.1, 0.2] for _ in texts], [{1: 0.5} for _ in texts]

            def increment_add_documents(self, texts):
                self.last_incremented = list(texts)

        class FakeMilvusManager:
            def __init__(self):
                self.insert_calls = 0

            def init_collection(self):
                return None

            def insert(self, rows):
                self.insert_calls += 1
                self.last_rows = rows

        docs = [
            {"text": "alpha", "filename": "a.pdf", "file_type": "pdf"},
            {"text": "beta", "filename": "b.pdf", "file_type": "pdf"},
        ]
        fake_cache = FakeCache()
        manager = FakeMilvusManager()
        writer = milvus_writer.MilvusWriter(embedding_service=FakeEmbeddingService(), milvus_manager=manager)

        with patch.object(milvus_writer, "cache", fake_cache):
            writer.write_documents(docs, batch_size=1)

        self.assertEqual(manager.insert_calls, 2)
        self.assertEqual(fake_cache.incr_keys, ["milvus_index_version"])

    def test_delete_and_drop_bump_index_version_after_success(self):
        fake_cache = FakeCache()
        manager = milvus_client.MilvusManager()

        with (
            patch.object(milvus_client, "cache", fake_cache),
            patch.object(manager, "_call_with_reconnect", return_value={"ok": True}),
        ):
            self.assertEqual(manager.delete('filename == "a.pdf"'), {"ok": True})
            manager.drop_collection()

        self.assertEqual(fake_cache.incr_keys, ["milvus_index_version", "milvus_index_version"])


if __name__ == "__main__":
    unittest.main()
