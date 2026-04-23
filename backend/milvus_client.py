"""Milvus client management for dense and sparse hybrid retrieval."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

load_dotenv()

logger = logging.getLogger(__name__)

QUERY_MAX_LIMIT = 16384

RECOVERABLE_MILVUS_ERROR_SNIPPETS = (
    "Cannot invoke RPC on closed channel",
    "failed to connect to all addresses",
    "connection refused",
    "deadline exceeded",
    "statuscode.unavailable",
    "transport is closing",
)


class MilvusManager:
    """Manage Milvus collection access and recover from stale RPC channels.

    Design note:
    - Do NOT keep a shared MilvusClient on the instance.
    - Create a fresh client per operation attempt.
    - Close it in finally.
    This avoids cross-thread / cross-request client invalidation.
    """

    def __init__(self) -> None:
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        self.uri = os.getenv("MILVUS_URI", f"http://{self.host}:{self.port}")

    def _new_client(self) -> MilvusClient:
        client = MilvusClient(uri=self.uri)
        logger.debug("Created Milvus client id=%s uri=%s", id(client), self.uri)
        return client

    @staticmethod
    def _close_client(client: MilvusClient | None) -> None:
        if client is None:
            return
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close Milvus client id=%s", id(client))

    @staticmethod
    def _is_recoverable_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(snippet.lower() in message for snippet in RECOVERABLE_MILVUS_ERROR_SNIPPETS)

    def _call_with_reconnect(
        self,
        operation: Callable[[MilvusClient], Any],
        retries: int = 2,
        operation_name: str = "milvus_operation",
    ) -> Any:
        """Execute an operation with fresh client creation on each attempt."""
        backoff = float(os.getenv("MILVUS_RECONNECT_BACKOFF_SECONDS", "0.05"))

        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            client: MilvusClient | None = None
            try:
                client = self._new_client()
                logger.debug(
                    "Milvus op=%s attempt=%s client_id=%s",
                    operation_name,
                    attempt,
                    id(client),
                )
                return operation(client)

            except Exception as exc:
                last_exc = exc
                recoverable = self._is_recoverable_error(exc)
                if not recoverable or attempt >= retries:
                    logger.warning(
                        "Milvus op=%s attempt=%s failed recoverable=%s error=%r",
                        operation_name,
                        attempt,
                        recoverable,
                        exc,
                    )
                    raise

                logger.info(
                    "Milvus op=%s attempt=%s failed recoverable=%s error=%r",
                    operation_name,
                    attempt,
                    recoverable,
                    exc,
                )

                if backoff > 0:
                    time.sleep(backoff * (attempt + 1))

            finally:
                self._close_client(client)

        raise RuntimeError("unreachable Milvus reconnect state") from last_exc

    def init_collection(self, dense_dim: int | None = None) -> None:
        """Initialize the Milvus collection if it does not already exist."""
        if dense_dim is None:
            dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))

        def operation(client: MilvusClient) -> None:
            if client.has_collection(self.collection_name):
                return

            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
            schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("text", DataType.VARCHAR, max_length=2000)
            schema.add_field("retrieval_text", DataType.VARCHAR, max_length=4000)
            schema.add_field("filename", DataType.VARCHAR, max_length=255)
            schema.add_field("file_type", DataType.VARCHAR, max_length=50)
            schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
            schema.add_field("page_number", DataType.INT64)
            schema.add_field("page_start", DataType.INT64)
            schema.add_field("page_end", DataType.INT64)
            schema.add_field("chunk_idx", DataType.INT64)
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("chunk_level", DataType.INT64)
            schema.add_field("chunk_role", DataType.VARCHAR, max_length=32)

            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="dense_embedding",
                index_type="HNSW",
                metric_type="IP",
                params={"M": 16, "efConstruction": 256},
            )
            index_params.add_index(
                field_name="sparse_embedding",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"drop_ratio_build": 0.2},
            )

            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )

        self._call_with_reconnect(operation, operation_name="init_collection")

    def insert(self, data: list[dict]) -> Any:
        return self._call_with_reconnect(
            lambda client: client.insert(self.collection_name, data),
            operation_name="insert",
        )

    def query(
        self,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
        limit: int = 10000,
        offset: int = 0,
    ) -> Any:
        safe_limit = max(0, min(limit, QUERY_MAX_LIMIT))
        safe_offset = max(0, offset)

        return self._call_with_reconnect(
            lambda client: client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields or ["filename", "file_type"],
                limit=safe_limit,
                offset=safe_offset,
            ),
            operation_name="query",
        )

    def query_all(
        self,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
    ) -> list[dict]:
        fields = output_fields or ["filename", "file_type"]
        out: list[dict] = []
        offset = 0

        while True:
            batch = self._call_with_reconnect(
                lambda client: client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=fields,
                    limit=QUERY_MAX_LIMIT,
                    offset=offset,
                ),
                operation_name="query_all",
            )

            if not batch:
                break

            out.extend(batch)

            if len(batch) < QUERY_MAX_LIMIT:
                break

            offset += len(batch)

        return out

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []

        quoted_ids = ", ".join([f'"{item}"' for item in ids])
        filter_expr = f"chunk_id in [{quoted_ids}]"

        return self.query(
            filter_expr=filter_expr,
            output_fields=[
                "text",
                "retrieval_text",
                "filename",
                "file_type",
                "page_number",
                "page_start",
                "page_end",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_role",
                "chunk_idx",
            ],
            limit=len(ids),
        )

    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict,
        top_k: int = 5,
        rrf_k: int = 60,
        filter_expr: str = "",
    ) -> list[dict]:
        output_fields = [
            "text",
            "retrieval_text",
            "filename",
            "file_type",
            "page_number",
            "page_start",
            "page_end",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_role",
            "chunk_idx",
            "section_title",
            "section_type",
            "section_path",
            "anchor_id",
        ]

        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=max(1, top_k * 2),
            expr=filter_expr,
        )
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=max(1, top_k * 2),
            expr=filter_expr,
        )
        reranker = RRFRanker(k=rrf_k)

        results = self._call_with_reconnect(
            lambda client: client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_search, sparse_search],
                ranker=reranker,
                limit=top_k,
                output_fields=output_fields,
            ),
            operation_name="hybrid_search",
        )

        formatted_results: list[dict] = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {
                        "id": hit.get("id"),
                        "text": hit.get("text", ""),
                        "retrieval_text": hit.get("retrieval_text", ""),
                        "filename": hit.get("filename", ""),
                        "file_type": hit.get("file_type", ""),
                        "page_number": hit.get("page_number", 0),
                        "page_start": hit.get("page_start", hit.get("page_number", 0)),
                        "page_end": hit.get("page_end", hit.get("page_number", 0)),
                        "chunk_id": hit.get("chunk_id", ""),
                        "parent_chunk_id": hit.get("parent_chunk_id", ""),
                        "root_chunk_id": hit.get("root_chunk_id", ""),
                        "chunk_level": hit.get("chunk_level", 0),
                        "chunk_role": hit.get("chunk_role", ""),
                        "chunk_idx": hit.get("chunk_idx", 0),
                        "section_title": hit.get("section_title", ""),
                        "section_type": hit.get("section_type", ""),
                        "section_path": hit.get("section_path", ""),
                        "anchor_id": hit.get("anchor_id", ""),
                        "score": hit.get("distance", 0.0),
                    }
                )

        return formatted_results

    def dense_retrieve(
        self,
        dense_embedding: list[float],
        top_k: int = 5,
        filter_expr: str = "",
    ) -> list[dict]:
        results = self._call_with_reconnect(
            lambda client: client.search(
                collection_name=self.collection_name,
                data=[dense_embedding],
                anns_field="dense_embedding",
                search_params={"metric_type": "IP", "params": {"ef": 64}},
                limit=top_k,
                output_fields=[
                    "text",
                    "retrieval_text",
                    "filename",
                    "file_type",
                    "page_number",
                    "page_start",
                    "page_end",
                    "chunk_id",
                    "parent_chunk_id",
                    "root_chunk_id",
                    "chunk_level",
                    "chunk_role",
                    "chunk_idx",
                    "section_title",
                    "section_type",
                    "section_path",
                    "anchor_id",
                ],
                filter=filter_expr,
            ),
            operation_name="dense_search",
        )

        formatted_results: list[dict] = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                formatted_results.append(
                    {
                        "id": hit.get("id"),
                        "text": entity.get("text", ""),
                        "retrieval_text": entity.get("retrieval_text", ""),
                        "filename": entity.get("filename", ""),
                        "file_type": entity.get("file_type", ""),
                        "page_number": entity.get("page_number", 0),
                        "page_start": entity.get("page_start", entity.get("page_number", 0)),
                        "page_end": entity.get("page_end", entity.get("page_number", 0)),
                        "chunk_id": entity.get("chunk_id", ""),
                        "parent_chunk_id": entity.get("parent_chunk_id", ""),
                        "root_chunk_id": entity.get("root_chunk_id", ""),
                        "chunk_level": entity.get("chunk_level", 0),
                        "chunk_role": entity.get("chunk_role", ""),
                        "chunk_idx": entity.get("chunk_idx", 0),
                        "section_title": entity.get("section_title", ""),
                        "section_type": entity.get("section_type", ""),
                        "section_path": entity.get("section_path", ""),
                        "anchor_id": entity.get("anchor_id", ""),
                        "score": hit.get("distance", 0.0),
                    }
                )

        return formatted_results

    def delete(self, filter_expr: str) -> Any:
        return self._call_with_reconnect(
            lambda client: client.delete(
                collection_name=self.collection_name,
                filter=filter_expr,
            ),
            operation_name="delete",
        )

    def has_collection(self) -> bool:
        return self._call_with_reconnect(
            lambda client: client.has_collection(self.collection_name),
            operation_name="has_collection",
        )

    def drop_collection(self) -> None:
        def operation(client: MilvusClient) -> None:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)

        self._call_with_reconnect(operation, operation_name="drop_collection")
