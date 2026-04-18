import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from sqlalchemy import func

from cache import cache as default_cache
from database import SessionLocal
from models import ChatMessage, ChatSession, User

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ConversationStorage:
    """Conversation storage backed by PostgreSQL with Redis read-through caching."""

    def __init__(self, session_factory=SessionLocal, cache_client=default_cache):
        self.session_factory = session_factory
        self.cache = cache_client

    @staticmethod
    def _messages_cache_key(user_id: str, session_id: str) -> str:
        return f"chat_messages:{user_id}:{session_id}"

    @staticmethod
    def _sessions_cache_key(user_id: str) -> str:
        return f"chat_sessions:{user_id}"

    @staticmethod
    def _to_langchain_messages(records: list[dict]) -> list:
        messages = []
        for msg_data in records:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            rag_trace = msg_data.get("rag_trace")
            if msg_type == "human":
                msg = HumanMessage(content=content)
            elif msg_type == "ai":
                msg = AIMessage(content=content)
            elif msg_type == "system":
                msg = SystemMessage(content=content)
            else:
                msg = HumanMessage(content=content)
            if rag_trace:
                msg.additional_kwargs["rag_trace"] = rag_trace
            messages.append(msg)
        return messages

    @staticmethod
    def _message_rag_trace(extra_message_data: list | None, idx: int) -> Any:
        if not extra_message_data or idx >= len(extra_message_data):
            return None
        extra = extra_message_data[idx] or {}
        return extra.get("rag_trace")

    @staticmethod
    def _serialize_message(message, timestamp: datetime, rag_trace=None) -> dict:
        return {
            "type": message.type,
            "content": str(message.content),
            "timestamp": timestamp.isoformat(),
            "rag_trace": rag_trace,
        }

    @staticmethod
    def _normalize_title_text(text: str) -> str:
        cleaned = " ".join((text or "").strip().split())
        return cleaned.strip("，。！？,.!?;；:：-—_")

    @classmethod
    def _build_session_title(cls, text: str) -> str | None:
        cleaned = cls._normalize_title_text(text)
        if not cleaned:
            return None
        max_len = 18
        return cleaned if len(cleaned) <= max_len else f"{cleaned[:max_len]}..."

    @classmethod
    def _extract_title_from_messages(cls, messages: list) -> str | None:
        for message in messages:
            if getattr(message, "type", "") == "human":
                return cls._build_session_title(str(getattr(message, "content", "")))
        return None

    def _get_user(self, db, user_id: str) -> User | None:
        return db.query(User).filter(User.username == user_id).first()

    def _get_or_create_session(self, db, user: User, session_id: str, metadata: dict | None = None) -> ChatSession:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.user_id == user.id, ChatSession.session_id == session_id)
            .first()
        )
        if session:
            if metadata is not None:
                session.metadata_json = metadata
            return session

        session = ChatSession(user_id=user.id, session_id=session_id, metadata_json=metadata or {})
        db.add(session)
        db.flush()
        return session

    def save(
        self,
        user_id: str,
        session_id: str,
        messages: list,
        metadata: dict | None = None,
        extra_message_data: list | None = None,
    ) -> None:
        """Replace a session's messages, used when old history is compacted."""
        db = self.session_factory()
        try:
            user = self._get_user(db, user_id)
            if not user:
                return

            session = self._get_or_create_session(db, user, session_id, metadata)
            db.query(ChatMessage).filter(ChatMessage.session_ref_id == session.id).delete(synchronize_session=False)

            now = utc_now()
            rows = []
            serialized = []
            for idx, message in enumerate(messages):
                rag_trace = self._message_rag_trace(extra_message_data, idx)
                rows.append(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=message.type,
                        content=str(message.content),
                        timestamp=now,
                        rag_trace=rag_trace,
                    )
                )
                serialized.append(self._serialize_message(message, now, rag_trace))

            if rows:
                db.add_all(rows)
            if not (session.metadata_json or {}).get("title"):
                title = self._extract_title_from_messages(messages)
                if title:
                    session.metadata_json = {**(session.metadata_json or {}), "title": title}
            session.updated_at = now
            db.commit()

            self.cache.set_json(self._messages_cache_key(user_id, session_id), serialized)
            self.cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    def append_messages(
        self,
        user_id: str,
        session_id: str,
        messages: list,
        metadata: dict | None = None,
        extra_message_data: list | None = None,
    ) -> None:
        """Append new messages for the common chat path without rewriting old rows."""
        if not messages:
            return

        db = self.session_factory()
        try:
            user = self._get_user(db, user_id)
            if not user:
                return

            session = self._get_or_create_session(db, user, session_id, metadata)
            now = utc_now()
            rows = []
            serialized = []
            for idx, message in enumerate(messages):
                rag_trace = self._message_rag_trace(extra_message_data, idx)
                rows.append(
                    ChatMessage(
                        session_ref_id=session.id,
                        message_type=message.type,
                        content=str(message.content),
                        timestamp=now,
                        rag_trace=rag_trace,
                    )
                )
                serialized.append(self._serialize_message(message, now, rag_trace))

            db.add_all(rows)
            if not (session.metadata_json or {}).get("title"):
                title = self._extract_title_from_messages(messages)
                if title:
                    session.metadata_json = {**(session.metadata_json or {}), "title": title}
            session.updated_at = now
            db.commit()

            cache_key = self._messages_cache_key(user_id, session_id)
            cached = self.cache.get_json(cache_key)
            if isinstance(cached, list):
                self.cache.set_json(cache_key, cached + serialized)
            self.cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    def replace_last_assistant_message(
        self,
        user_id: str,
        session_id: str,
        ai_message,
        rag_trace=None,
    ) -> None:
        """删除会话中最后一条助手消息并写入新的助手回复（用于重新生成）。"""
        db = self.session_factory()
        try:
            user = self._get_user(db, user_id)
            if not user:
                return

            session = (
                db.query(ChatSession)
                .join(User, ChatSession.user_id == User.id)
                .filter(User.username == user_id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return

            last_msg = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_ref_id == session.id)
                .order_by(ChatMessage.id.desc())
                .first()
            )
            if last_msg and last_msg.message_type == "ai":
                db.delete(last_msg)
                db.flush()

            now = utc_now()
            db.add(
                ChatMessage(
                    session_ref_id=session.id,
                    message_type="ai",
                    content=str(ai_message.content),
                    timestamp=now,
                    rag_trace=rag_trace,
                )
            )
            session.updated_at = now
            db.commit()

            self.cache.delete(self._messages_cache_key(user_id, session_id))
            self.cache.delete(self._sessions_cache_key(user_id))
        finally:
            db.close()

    def load(self, user_id: str, session_id: str) -> list:
        cached = self.cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return self._to_langchain_messages(cached)

        records = self.get_session_messages(user_id, session_id)
        self.cache.set_json(self._messages_cache_key(user_id, session_id), records)
        return self._to_langchain_messages(records)

    def list_sessions(self, user_id: str) -> list:
        return [item["session_id"] for item in self.list_session_infos(user_id)]

    def list_session_infos(self, user_id: str) -> list[dict]:
        cached = self.cache.get_json(self._sessions_cache_key(user_id))
        if cached is not None:
            return cached

        db = self.session_factory()
        try:
            rows = (
                db.query(
                    ChatSession.session_id,
                    ChatSession.metadata_json,
                    ChatSession.updated_at,
                    func.count(ChatMessage.id).label("message_count"),
                )
                .join(User, ChatSession.user_id == User.id)
                .outerjoin(ChatMessage, ChatMessage.session_ref_id == ChatSession.id)
                .filter(User.username == user_id)
                .group_by(ChatSession.id)
                .order_by(ChatSession.updated_at.desc())
                .all()
            )
            result = [
                {
                    "session_id": row.session_id,
                    "title": (row.metadata_json or {}).get("title"),
                    "updated_at": row.updated_at.isoformat() if row.updated_at else "",
                    "message_count": int(row.message_count or 0),
                }
                for row in rows
            ]
            self.cache.set_json(self._sessions_cache_key(user_id), result)
            return result
        except Exception:
            logger.exception("Failed to load sessions for user=%s", user_id)
            return []
        finally:
            db.close()

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        cached = self.cache.get_json(self._messages_cache_key(user_id, session_id))
        if cached is not None:
            return cached

        db = self.session_factory()
        try:
            rows = (
                db.query(ChatMessage)
                .join(ChatSession, ChatMessage.session_ref_id == ChatSession.id)
                .join(User, ChatSession.user_id == User.id)
                .filter(User.username == user_id, ChatSession.session_id == session_id)
                .order_by(ChatMessage.id.asc())
                .all()
            )
            result = [
                {
                    "type": row.message_type,
                    "content": row.content,
                    "timestamp": row.timestamp.isoformat(),
                    "rag_trace": row.rag_trace,
                }
                for row in rows
            ]
            self.cache.set_json(self._messages_cache_key(user_id, session_id), result)
            return result
        finally:
            db.close()

    def update_session_title(self, user_id: str, session_id: str, title: str | None) -> dict | None:
        """设置或清空会话标题。空字符串表示清除自定义标题。"""
        db = self.session_factory()
        try:
            session = (
                db.query(ChatSession)
                .join(User, ChatSession.user_id == User.id)
                .filter(User.username == user_id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return None

            meta = dict(session.metadata_json or {})
            raw = (title or "").strip()
            if not raw:
                meta.pop("title", None)
            else:
                normalized = self._normalize_title_text(raw)
                if not normalized:
                    meta.pop("title", None)
                else:
                    meta["title"] = normalized[:80]
            session.metadata_json = meta
            session.updated_at = utc_now()
            db.commit()
            self.cache.delete(self._sessions_cache_key(user_id))
            return {
                "session_id": session_id,
                "title": meta.get("title"),
            }
        finally:
            db.close()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        db = self.session_factory()
        try:
            session = (
                db.query(ChatSession)
                .join(User, ChatSession.user_id == User.id)
                .filter(User.username == user_id, ChatSession.session_id == session_id)
                .first()
            )
            if not session:
                return False

            db.delete(session)
            db.commit()
            self.cache.delete(self._messages_cache_key(user_id, session_id))
            self.cache.delete(self._sessions_cache_key(user_id))
            return True
        finally:
            db.close()
