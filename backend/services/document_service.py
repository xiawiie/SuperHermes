"""Service layer for document operations, decoupled from FastAPI protocol."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from backend.documents.loader import DocumentLoader
from backend.infra.vector_store.milvus_client import MilvusManager
from backend.infra.vector_store.milvus_writer import MilvusWriter
from backend.infra.embedding import embedding_service
from backend.infra.vector_store.parent_chunk_store import ParentChunkStore


class DocumentProcessingError(RuntimeError):
    """Raised when source content cannot be converted into searchable chunks."""


class DocumentService:
    """Encapsulates document ingestion, listing, and deletion logic."""

    def __init__(
        self,
        loader: DocumentLoader,
        milvus_manager: MilvusManager,
        milvus_writer: MilvusWriter,
        parent_store: ParentChunkStore,
        upload_dir: Path,
        embedding,
    ):
        self._loader = loader
        self._milvus = milvus_manager
        self._writer = milvus_writer
        self._parent = parent_store
        self._upload_dir = upload_dir
        self._embedding = embedding

    @classmethod
    def create_default(cls, upload_dir: Path | None = None) -> "DocumentService":
        loader = DocumentLoader()
        milvus = MilvusManager()
        writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=milvus)
        parent = ParentChunkStore()
        base = upload_dir or (Path(__file__).resolve().parents[1] / "data" / "documents")
        return cls(loader, milvus, writer, parent, base, embedding_service)

    def list_documents(self) -> list[dict[str, Any]]:
        self._milvus.init_collection()
        results = self._milvus.query(output_fields=["filename", "file_type"], limit=10000)
        stats: dict[str, dict] = {}
        for item in results:
            fn = item.get("filename", "")
            ft = item.get("file_type", "")
            if fn not in stats:
                stats[fn] = {"filename": fn, "file_type": ft, "chunk_count": 0}
            stats[fn]["chunk_count"] += 1
        return list(stats.values())

    def _remove_bm25_stats(self, filename: str) -> None:
        rows = self._milvus.query_all(
            filter_expr=f'filename == "{filename}"',
            output_fields=["text"],
        )
        texts = [r.get("text") or "" for r in rows]
        self._embedding.increment_remove_documents(texts)

    def upload_document(self, filename: str, content: bytes) -> dict[str, Any]:
        os.makedirs(self._upload_dir, exist_ok=True)
        self._milvus.init_collection()

        delete_expr = f'filename == "{filename}"'
        try:
            self._remove_bm25_stats(filename)
        except Exception:
            pass
        try:
            self._milvus.delete(delete_expr)
        except Exception:
            pass
        try:
            self._parent.delete_by_filename(filename)
        except Exception:
            pass

        file_path = self._upload_dir / filename
        file_path.write_bytes(content)

        try:
            new_docs = self._loader.load_document(str(file_path), filename)
        except Exception as doc_err:
            raise DocumentProcessingError(f"Failed to load document: {doc_err}") from doc_err
        if not new_docs:
            raise DocumentProcessingError("Document processing failed: no content extracted")

        parent_docs = [d for d in new_docs if int(d.get("chunk_level", 0) or 0) in (1, 2)]
        leaf_docs = [d for d in new_docs if int(d.get("chunk_level", 0) or 0) == 3]
        if not leaf_docs:
            raise DocumentProcessingError("Document processing failed: no leaf chunks generated")

        self._parent.upsert_documents(parent_docs)
        self._writer.write_documents(leaf_docs)

        return {
            "filename": filename,
            "chunks_processed": len(leaf_docs),
            "message": (
                f"Uploaded {filename}; indexed {len(leaf_docs)} leaf chunks "
                f"and stored {len(parent_docs)} parent chunks"
            ),
        }

    def delete_document(self, filename: str) -> dict[str, Any]:
        self._milvus.init_collection()
        delete_expr = f'filename == "{filename}"'
        self._remove_bm25_stats(filename)
        result = self._milvus.delete(delete_expr)
        self._parent.delete_by_filename(filename)
        return {
            "filename": filename,
            "chunks_deleted": result.get("delete_count", 0) if isinstance(result, dict) else 0,
            "message": f"Deleted document {filename} from vector store and parent chunk storage",
        }
