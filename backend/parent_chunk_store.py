"""Parent chunk storage for auto-merging retrieval."""
from datetime import datetime, timezone, timedelta
from typing import List

from sqlalchemy.dialects.postgresql import insert as pg_insert

from cache import cache
from database import SessionLocal
from models import ParentChunk
from rag_profiles import (
    LEGACY_INDEX_PROFILE,
    current_index_profile,
    display_chunk_id,
    normalize_index_profile,
    storage_chunk_id,
)

_BJ_TZ = timezone(timedelta(hours=8))


def local_now() -> datetime:
    """返回当前北京时间（UTC+8，不带时区信息）。"""
    return datetime.now(_BJ_TZ).replace(tzinfo=None)


class ParentChunkStore:
    """PostgreSQL + Redis storage for parent chunks."""

    def __init__(self, index_profile: str | None = None) -> None:
        self.index_profile = normalize_index_profile(index_profile or current_index_profile())

    def _to_dict(self, item: ParentChunk) -> dict:
        return {
            "text": item.text,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "chunk_id": display_chunk_id(item.chunk_id, self.index_profile),
            "parent_chunk_id": item.parent_chunk_id,
            "root_chunk_id": item.root_chunk_id,
            "chunk_level": item.chunk_level,
            "chunk_idx": item.chunk_idx,
            "index_profile": getattr(item, "index_profile", self.index_profile) or self.index_profile,
        }

    def _cache_key(self, chunk_id: str) -> str:
        if self.index_profile == LEGACY_INDEX_PROFILE:
            return f"parent_chunk:{chunk_id}"
        return f"parent_chunk:{self.index_profile}:{chunk_id}"

    def _payload_from_doc(self, doc: dict, updated_at: datetime) -> dict:
        chunk_id = (doc.get("chunk_id") or "").strip()
        return {
            "chunk_id": storage_chunk_id(chunk_id, self.index_profile),
            "index_profile": self.index_profile,
            "text": doc.get("text", ""),
            "filename": doc.get("filename", ""),
            "file_type": doc.get("file_type", ""),
            "file_path": doc.get("file_path", ""),
            "page_number": int(doc.get("page_number", 0) or 0),
            "parent_chunk_id": doc.get("parent_chunk_id", ""),
            "root_chunk_id": doc.get("root_chunk_id", ""),
            "chunk_level": int(doc.get("chunk_level", 0) or 0),
            "chunk_idx": int(doc.get("chunk_idx", 0) or 0),
            "updated_at": updated_at,
        }

    def _cache_payload(self, payload: dict) -> dict:
        return {
            "chunk_id": display_chunk_id(payload["chunk_id"], self.index_profile),
            "text": payload["text"],
            "filename": payload["filename"],
            "file_type": payload["file_type"],
            "file_path": payload["file_path"],
            "page_number": payload["page_number"],
            "parent_chunk_id": payload["parent_chunk_id"],
            "root_chunk_id": payload["root_chunk_id"],
            "chunk_level": payload["chunk_level"],
            "chunk_idx": payload["chunk_idx"],
            "index_profile": payload["index_profile"],
        }

    def upsert_documents(self, docs: List[dict]) -> int:
        """Insert or update parent chunks and return the number of written rows."""
        if not docs:
            return 0

        now = local_now()
        rows = []
        cache_payloads = {}
        seen = set()
        for doc in docs:
            payload = self._payload_from_doc(doc, now)
            original_chunk_id = (doc.get("chunk_id") or "").strip()
            storage_id = payload["chunk_id"]
            if not original_chunk_id or storage_id in seen:
                continue
            seen.add(storage_id)
            rows.append(payload)
            cache_payloads[self._cache_key(original_chunk_id)] = self._cache_payload(payload)

        if not rows:
            return 0

        db = SessionLocal()
        try:
            if db.get_bind().dialect.name == "postgresql":
                stmt = pg_insert(ParentChunk).values(rows)
                update_columns = {
                    column.name: getattr(stmt.excluded, column.name)
                    for column in ParentChunk.__table__.columns
                    if column.name != "chunk_id"
                }
                db.execute(
                    stmt.on_conflict_do_update(
                        index_elements=[ParentChunk.chunk_id],
                        set_=update_columns,
                    )
                )
            else:
                for payload in rows:
                    record = db.get(ParentChunk, payload["chunk_id"])
                    if record:
                        for key, value in payload.items():
                            setattr(record, key, value)
                    else:
                        db.add(ParentChunk(**payload))
            db.commit()
        finally:
            db.close()

        cache.set_many_json(cache_payloads)
        return len(rows)

    def get_documents_by_ids(self, chunk_ids: List[str]) -> List[dict]:
        if not chunk_ids:
            return []

        ordered_results = {}
        normalized_ids = []
        for chunk_id in chunk_ids:
            key = (chunk_id or "").strip()
            if key:
                normalized_ids.append(key)

        cache_keys = {self._cache_key(chunk_id): chunk_id for chunk_id in normalized_ids}
        cached_items = cache.get_many_json(list(cache_keys.keys()))
        for cache_key, payload in cached_items.items():
            ordered_results[cache_keys[cache_key]] = payload

        missing_ids = [chunk_id for chunk_id in normalized_ids if chunk_id not in ordered_results]
        if missing_ids:
            storage_ids = [storage_chunk_id(chunk_id, self.index_profile) for chunk_id in missing_ids]
            db = SessionLocal()
            try:
                rows = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(storage_ids)).all()
                cache_updates = {}
                for row in rows:
                    payload = self._to_dict(row)
                    original_id = payload["chunk_id"]
                    ordered_results[original_id] = payload
                    cache_updates[self._cache_key(original_id)] = payload
                cache.set_many_json(cache_updates)
            finally:
                db.close()

        return [ordered_results[item] for item in chunk_ids if item in ordered_results]

    def delete_by_filename(self, filename: str) -> int:
        """Delete parent chunks by source filename and return the deleted count."""
        if not filename:
            return 0

        db = SessionLocal()
        try:
            rows = (
                db.query(ParentChunk)
                .filter(ParentChunk.filename == filename, ParentChunk.index_profile == self.index_profile)
                .all()
            )
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            if deleted > 0:
                db.query(ParentChunk).filter(
                    ParentChunk.filename == filename,
                    ParentChunk.index_profile == self.index_profile,
                ).delete(synchronize_session=False)
                db.commit()
                cache.delete_many([self._cache_key(display_chunk_id(chunk_id, self.index_profile)) for chunk_id in chunk_ids])
            return deleted
        finally:
            db.close()

    def delete_by_profile(self) -> int:
        """Delete all parent chunks for this profile and return the deleted count."""
        db = SessionLocal()
        try:
            rows = db.query(ParentChunk).filter(ParentChunk.index_profile == self.index_profile).all()
            original_ids = [display_chunk_id(row.chunk_id, self.index_profile) for row in rows]
            deleted = len(original_ids)
            if deleted:
                db.query(ParentChunk).filter(ParentChunk.index_profile == self.index_profile).delete(
                    synchronize_session=False
                )
                db.commit()
                cache.delete_many([self._cache_key(chunk_id) for chunk_id in original_ids])
            return deleted
        finally:
            db.close()
