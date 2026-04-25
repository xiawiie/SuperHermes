"""Tests for BM25 state isolation per collection and text mode."""
from __future__ import annotations

import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


class TestBM25StateIsolation:
    def test_default_state_path_includes_collection(self):
        collection = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        text_mode = os.getenv("EVAL_RETRIEVAL_TEXT_MODE", "title_context")
        from embedding import _DEFAULT_STATE_PATH
        expected_name = f"bm25_state_{collection}_{text_mode}.json"
        assert _DEFAULT_STATE_PATH.name == expected_name

    def test_different_collection_different_path(self):
        """Verify that changing MILVUS_COLLECTION changes new service state paths."""
        from embedding import EmbeddingService

        old_collection = os.environ.get("MILVUS_COLLECTION")
        old_text_mode = os.environ.get("EVAL_RETRIEVAL_TEXT_MODE")
        old_state_path = os.environ.pop("BM25_STATE_PATH", None)
        try:
            os.environ["MILVUS_COLLECTION"] = "collection_a"
            os.environ["EVAL_RETRIEVAL_TEXT_MODE"] = "title_context"
            svc_a = EmbeddingService()

            os.environ["MILVUS_COLLECTION"] = "collection_b"
            os.environ["EVAL_RETRIEVAL_TEXT_MODE"] = "title_context_filename"
            svc_b = EmbeddingService()

            assert svc_a._state_path.name == "bm25_state_collection_a_title_context.json"
            assert svc_b._state_path.name == "bm25_state_collection_b_title_context_filename.json"
            assert svc_a._state_path != svc_b._state_path
        finally:
            if old_collection is not None:
                os.environ["MILVUS_COLLECTION"] = old_collection
            else:
                os.environ.pop("MILVUS_COLLECTION", None)
            if old_text_mode is not None:
                os.environ["EVAL_RETRIEVAL_TEXT_MODE"] = old_text_mode
            else:
                os.environ.pop("EVAL_RETRIEVAL_TEXT_MODE", None)
            if old_state_path is not None:
                os.environ["BM25_STATE_PATH"] = old_state_path

    def test_state_path_format(self):
        """Verify state path follows bm25_state_{collection}_{mode}.json pattern."""
        from embedding import _DEFAULT_STATE_PATH
        name = _DEFAULT_STATE_PATH.name
        assert name.startswith("bm25_state_")
        assert name.endswith(".json")
        parts = name.replace("bm25_state_", "").replace(".json", "").split("_")
        # Should have at least collection + mode parts
        assert len(parts) >= 2

    def test_custom_state_path_via_env(self):
        """Verify BM25_STATE_PATH env overrides default."""
        custom_path = os.path.join(os.path.dirname(__file__), "custom_bm25_state.json")
        old = os.environ.get("BM25_STATE_PATH")
        try:
            os.environ["BM25_STATE_PATH"] = custom_path
            from embedding import EmbeddingService
            svc = EmbeddingService()
            # On Windows, Path normalizes slashes; compare Path objects
            assert str(svc._state_path).replace("\\", "/") == custom_path.replace("\\", "/")
        finally:
            if old is not None:
                os.environ["BM25_STATE_PATH"] = old
            else:
                os.environ.pop("BM25_STATE_PATH", None)
