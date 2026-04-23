"""Parent chunk storage for auto-merging retrieval."""
from datetime import datetime, timezone, timedelta
from typing import List

from sqlalchemy.dialects.postgresql import insert as pg_insert

from cache import cache
from database import SessionLocal
from models import ParentChunk

_BJ_TZ = timezone(timedelta(hours=8))


def local_now() -> datetime:
    """返回当前北京时间（UTC+8，不带时区信息）。"""
    return datetime.now(_BJ_TZ).replace(tzinfo=None)


class ParentChunkStore:
    """PostgreSQL + Redis storage for parent chunks."""

    @staticmethod
    def _to_dict(item: ParentChunk) -> dict:
        return {
            "text": item.text,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "chunk_id": item.chunk_id,
            "parent_chunk_id": item.parent_chunk_id,
            "root_chunk_id": item.root_chunk_id,
            "chunk_level": item.chunk_level,
            "chunk_idx": item.chunk_idx,
        }

    @staticmethod
    def _cache_key(chunk_id: str) -> str:
        return f"parent_chunk:{chunk_id}"

    @staticmethod
    def _payload_from_doc(doc: dict, updated_at: datetime) -> dict:
        return {
            "chunk_id": (doc.get("chunk_id") or "").strip(),
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

    @staticmethod
    def _cache_payload(payload: dict) -> dict:
        return {
            "chunk_id": payload["chunk_id"],
            "text": payload["text"],
            "filename": payload["filename"],
            "file_type": payload["file_type"],
            "file_path": payload["file_path"],
            "page_number": payload["page_number"],
            "parent_chunk_id": payload["parent_chunk_id"],
            "root_chunk_id": payload["root_chunk_id"],
            "chunk_level": payload["chunk_level"],
            "chunk_idx": payload["chunk_idx"],
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
            chunk_id = payload["chunk_id"]
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            rows.append(payload)
            cache_payloads[self._cache_key(chunk_id)] = self._cache_payload(payload)

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
            db = SessionLocal()
            try:
                rows = db.query(ParentChunk).filter(ParentChunk.chunk_id.in_(missing_ids)).all()
                cache_updates = {}
                for row in rows:
                    payload = self._to_dict(row)
                    ordered_results[row.chunk_id] = payload
                    cache_updates[self._cache_key(row.chunk_id)] = payload
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
            rows = db.query(ParentChunk).filter(ParentChunk.filename == filename).all()
            chunk_ids = [row.chunk_id for row in rows]
            deleted = len(chunk_ids)
            if deleted > 0:
                db.query(ParentChunk).filter(ParentChunk.filename == filename).delete(synchronize_session=False)
                db.commit()
                cache.delete_many([self._cache_key(chunk_id) for chunk_id in chunk_ids])
            return deleted
        finally:
            db.close()
