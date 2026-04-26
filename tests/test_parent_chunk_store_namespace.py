import sys
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from database import Base  # noqa: E402
from models import ParentChunk  # noqa: E402
import parent_chunk_store as store_module  # noqa: E402
from parent_chunk_store import ParentChunkStore  # noqa: E402


class MemoryCache:
    def __init__(self):
        self.values = {}
        self.deleted = []

    def set_many_json(self, mapping):
        self.values.update(mapping)

    def get_many_json(self, keys):
        return {key: self.values[key] for key in keys if key in self.values}

    def delete_many(self, keys):
        self.deleted.extend(keys)
        for key in keys:
            self.values.pop(key, None)

    def delete_pattern(self, pattern):
        prefix = pattern.rstrip("*")
        for key in list(self.values):
            if key.startswith(prefix):
                self.values.pop(key, None)


class ParentChunkStoreNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False)
        self.old_session = store_module.SessionLocal
        self.old_cache = store_module.cache
        self.cache = MemoryCache()
        store_module.SessionLocal = self.Session
        store_module.cache = self.cache

    def tearDown(self):
        store_module.SessionLocal = self.old_session
        store_module.cache = self.old_cache

    def test_profiles_can_store_same_chunk_id_without_cross_read(self):
        doc_a = {
            "chunk_id": "parent-1",
            "text": "quality parent",
            "filename": "manual.pdf",
            "file_type": "pdf",
            "page_number": 1,
            "chunk_level": 1,
        }
        doc_b = {
            **doc_a,
            "text": "fast parent",
        }

        quality = ParentChunkStore(index_profile="v3_quality")
        fast = ParentChunkStore(index_profile="v3_fast")
        quality.upsert_documents([doc_a])
        fast.upsert_documents([doc_b])

        self.assertEqual(quality.get_documents_by_ids(["parent-1"])[0]["text"], "quality parent")
        self.assertEqual(fast.get_documents_by_ids(["parent-1"])[0]["text"], "fast parent")

        db = self.Session()
        rows = db.query(ParentChunk).order_by(ParentChunk.index_profile.asc()).all()
        db.close()
        self.assertEqual(len(rows), 2)
        self.assertEqual({row.index_profile for row in rows}, {"v3_quality", "v3_fast"})
        self.assertTrue(all(row.chunk_id.endswith("::parent-1") for row in rows))

    def test_legacy_profile_keeps_unprefixed_chunk_id(self):
        legacy = ParentChunkStore(index_profile="legacy")
        legacy.upsert_documents([
            {
                "chunk_id": "parent-legacy",
                "text": "legacy parent",
                "filename": "manual.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_level": 1,
            }
        ])

        self.assertEqual(legacy.get_documents_by_ids(["parent-legacy"])[0]["chunk_id"], "parent-legacy")
        db = self.Session()
        row = db.get(ParentChunk, "parent-legacy")
        db.close()
        self.assertIsNotNone(row)
        self.assertEqual(row.index_profile, "legacy")


if __name__ == "__main__":
    unittest.main()
