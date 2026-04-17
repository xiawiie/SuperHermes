"""Milvus client management for dense and sparse hybrid retrieval."""
import os

from dotenv import load_dotenv
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

load_dotenv()

QUERY_MAX_LIMIT = 16384


class MilvusManager:
    """Manage Milvus collection access and recover from stale RPC channels."""

    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        self.uri = f"http://{self.host}:{self.port}"
        self.client = None

    def _get_client(self) -> MilvusClient:
        if self.client is None:
            self.client = MilvusClient(uri=self.uri)
        return self.client

    @staticmethod
    def _is_closed_channel_error(exc: Exception) -> bool:
        return "Cannot invoke RPC on closed channel" in str(exc)

    def _discard_client(self) -> None:
        client = self.client
        self.client = None
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _call_with_reconnect(self, operation):
        try:
            return operation(self._get_client())
        except Exception as exc:
            if not self._is_closed_channel_error(exc):
                raise
            self._discard_client()
            return operation(self._get_client())

    def init_collection(self, dense_dim: int | None = None):
        """Initialize the Milvus collection if it does not already exist."""
        if dense_dim is None:
            dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))

        def operation(client: MilvusClient):
            if client.has_collection(self.collection_name):
                return

            schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
            schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
            schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field("text", DataType.VARCHAR, max_length=2000)
            schema.add_field("filename", DataType.VARCHAR, max_length=255)
            schema.add_field("file_type", DataType.VARCHAR, max_length=50)
            schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
            schema.add_field("page_number", DataType.INT64)
            schema.add_field("chunk_idx", DataType.INT64)
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
            schema.add_field("chunk_level", DataType.INT64)

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

        return self._call_with_reconnect(operation)

    def insert(self, data: list[dict]):
        return self._call_with_reconnect(
            lambda client: client.insert(self.collection_name, data)
        )

    def query(
        self,
        filter_expr: str = "",
        output_fields: list[str] | None = None,
        limit: int = 10000,
        offset: int = 0,
    ):
        return self._call_with_reconnect(
            lambda client: client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields or ["filename", "file_type"],
                limit=min(limit, QUERY_MAX_LIMIT),
                offset=offset,
            )
        )

    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list:
        fields = output_fields or ["filename", "file_type"]
        out: list = []
        offset = 0
        while True:
            batch = self._call_with_reconnect(
                lambda client: client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=fields,
                    limit=QUERY_MAX_LIMIT,
                    offset=offset,
                )
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
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
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
            "filename",
            "file_type",
            "page_number",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_idx",
        ]

        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k * 2,
            expr=filter_expr,
        )
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,
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
            )
        )

        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append(
                    {
                        "id": hit.get("id"),
                        "text": hit.get("text", ""),
                        "filename": hit.get("filename", ""),
                        "file_type": hit.get("file_type", ""),
                        "page_number": hit.get("page_number", 0),
                        "chunk_id": hit.get("chunk_id", ""),
                        "parent_chunk_id": hit.get("parent_chunk_id", ""),
                        "root_chunk_id": hit.get("root_chunk_id", ""),
                        "chunk_level": hit.get("chunk_level", 0),
                        "chunk_idx": hit.get("chunk_idx", 0),
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
                    "filename",
                    "file_type",
                    "page_number",
                    "chunk_id",
                    "parent_chunk_id",
                    "root_chunk_id",
                    "chunk_level",
                    "chunk_idx",
                ],
                filter=filter_expr,
            )
        )

        formatted_results = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                formatted_results.append(
                    {
                        "id": hit.get("id"),
                        "text": entity.get("text", ""),
                        "filename": entity.get("filename", ""),
                        "file_type": entity.get("file_type", ""),
                        "page_number": entity.get("page_number", 0),
                        "chunk_id": entity.get("chunk_id", ""),
                        "parent_chunk_id": entity.get("parent_chunk_id", ""),
                        "root_chunk_id": entity.get("root_chunk_id", ""),
                        "chunk_level": entity.get("chunk_level", 0),
                        "chunk_idx": entity.get("chunk_idx", 0),
                        "score": hit.get("distance", 0.0),
                    }
                )
        return formatted_results

    def delete(self, filter_expr: str):
        return self._call_with_reconnect(
            lambda client: client.delete(
                collection_name=self.collection_name,
                filter=filter_expr,
            )
        )

    def has_collection(self) -> bool:
        return self._call_with_reconnect(
            lambda client: client.has_collection(self.collection_name)
        )

    def drop_collection(self):
        def operation(client: MilvusClient):
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)

        return self._call_with_reconnect(operation)
