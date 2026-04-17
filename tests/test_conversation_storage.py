import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from database import Base  # noqa: E402
from models import ChatMessage, ChatSession, User  # noqa: E402


class MemoryCache:
    def __init__(self):
        self.values = {}
        self.deleted = []

    def get_json(self, key):
        return self.values.get(key)

    def set_json(self, key, value, ttl=None):
        self.values[key] = value

    def get_many_json(self, keys):
        return {key: self.values[key] for key in keys if key in self.values}

    def set_many_json(self, mapping, ttl=None):
        self.values.update(mapping)

    def delete(self, key):
        self.deleted.append(key)
        self.values.pop(key, None)

    def delete_many(self, keys):
        for key in keys:
            self.delete(key)


def msg(message_type, content):
    return SimpleNamespace(type=message_type, content=content)


class ConversationStorageTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False)

        db = self.Session()
        db.add(User(username="alice", password_hash="x", role="user"))
        db.commit()
        db.close()

    def build_storage(self):
        from conversation_storage import ConversationStorage

        return ConversationStorage(session_factory=self.Session, cache_client=MemoryCache())

    def test_append_messages_preserves_existing_rows(self):
        storage = self.build_storage()

        storage.append_messages(
            "alice",
            "session-1",
            [msg("human", "hello"), msg("ai", "hi")],
            extra_message_data=[None, {"rag_trace": {"tool_used": False}}],
        )

        db = self.Session()
        first_ids = [row.id for row in db.query(ChatMessage).order_by(ChatMessage.id.asc()).all()]
        db.close()

        storage.append_messages(
            "alice",
            "session-1",
            [msg("human", "next"), msg("ai", "answer")],
            extra_message_data=[None, {"rag_trace": {"tool_used": True}}],
        )

        db = self.Session()
        rows = db.query(ChatMessage).order_by(ChatMessage.id.asc()).all()
        db.close()

        self.assertEqual(first_ids, [rows[0].id, rows[1].id])
        self.assertEqual([row.message_type for row in rows], ["human", "ai", "human", "ai"])
        self.assertEqual(rows[-1].rag_trace, {"tool_used": True})

    def test_list_session_infos_uses_aggregate_query_for_message_counts(self):
        storage = self.build_storage()
        storage.append_messages("alice", "older", [msg("human", "one"), msg("ai", "two")])
        storage.append_messages("alice", "newer", [msg("human", "one"), msg("ai", "two"), msg("human", "three")])

        statements = []

        @event.listens_for(self.engine, "before_cursor_execute")
        def capture_selects(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        infos = storage.list_session_infos("alice")

        event.remove(self.engine, "before_cursor_execute", capture_selects)

        counts = {item["session_id"]: item["message_count"] for item in infos}
        self.assertEqual(counts, {"older": 2, "newer": 3})
        self.assertLessEqual(len(statements), 1)

    def test_replace_messages_still_supports_compacted_history(self):
        storage = self.build_storage()
        storage.append_messages("alice", "session-1", [msg("human", "old"), msg("ai", "old answer")])

        storage.save(
            "alice",
            "session-1",
            [msg("system", "summary"), msg("human", "latest"), msg("ai", "latest answer")],
        )

        db = self.Session()
        rows = db.query(ChatMessage).join(ChatSession).filter(ChatSession.session_id == "session-1").order_by(ChatMessage.id.asc()).all()
        db.close()

        self.assertEqual([row.message_type for row in rows], ["system", "human", "ai"])
        self.assertEqual([row.content for row in rows], ["summary", "latest", "latest answer"])


if __name__ == "__main__":
    unittest.main()
