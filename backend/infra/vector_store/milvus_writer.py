"""Write chunk documents into Milvus with dense and sparse embeddings."""
from __future__ import annotations

from backend.infra.cache import cache
from backend.infra.embedding import EmbeddingService, embedding_service as _default_embedding_service
from backend.infra.vector_store.milvus_client import MilvusManager
from backend.rag.profiles import current_index_profile


class MilvusWriter:
    """Persist leaf chunks into Milvus."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        milvus_manager: MilvusManager | None = None,
    ) -> None:
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or MilvusManager()
        self.index_profile = current_index_profile()

    def write_documents(self, documents: list[dict], batch_size: int = 50) -> None:
        if not documents:
            return
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.milvus_manager.init_collection()

        total = len(documents)
        inserted_any = False
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            prepared_batch = self._prepare_batch(batch)
            if not prepared_batch:
                continue

            insert_data = [
                {
                    "dense_embedding": dense_emb,
                    "sparse_embedding": sparse_emb,
                    "text": doc["text"],
                    "retrieval_text": doc.get("retrieval_text", doc["text"]),
                    "filename": doc["filename"],
                    "file_type": doc["file_type"],
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "page_start": doc.get("page_start", doc.get("page_number", 0)),
                    "page_end": doc.get("page_end", doc.get("page_number", 0)),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                    "chunk_role": doc.get("chunk_role", ""),
                    "index_profile": doc.get("index_profile", self.index_profile),
                    "section_title": doc.get("section_title", ""),
                    "section_type": doc.get("section_type", ""),
                    "section_path": doc.get("section_path", ""),
                    "anchor_id": doc.get("anchor_id", ""),
                }
                for doc, dense_emb, sparse_emb in prepared_batch
            ]

            self.milvus_manager.insert(insert_data)
            inserted_any = True

        if inserted_any:
            cache.incr("milvus_index_version")

    def _prepare_batch(self, batch: list[dict]) -> list[tuple[dict, list[float], dict]]:
        texts = [doc.get("retrieval_text") or doc["text"] for doc in batch]
        try:
            dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)
            self.embedding_service.increment_add_documents(texts)
            return list(zip(batch, dense_embeddings, sparse_embeddings))
        except Exception:
            prepared: list[tuple[dict, list[float], dict]] = []
            successful_texts: list[str] = []
            for doc, text in zip(batch, texts):
                try:
                    dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings([text])
                except Exception:
                    continue
                prepared.append((doc, dense_embeddings[0], sparse_embeddings[0]))
                successful_texts.append(text)
            if successful_texts:
                self.embedding_service.increment_add_documents(successful_texts)
            return prepared
