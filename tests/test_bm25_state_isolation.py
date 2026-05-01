"""Tests for BM25 state isolation per collection and text mode."""
from __future__ import annotations

import os

import pytest


class TestBM25StateIsolation:
    def test_default_state_path_includes_collection(self):
        collection = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        text_mode = os.getenv("EVAL_RETRIEVAL_TEXT_MODE", "title_context")
        from backend.infra.embedding import _DEFAULT_STATE_PATH
        expected_name = f"bm25_state_{collection}_{text_mode}.json"
        assert _DEFAULT_STATE_PATH.name == expected_name

    def test_different_collection_different_path(self):
        """Verify that changing MILVUS_COLLECTION changes new service state paths."""
        from unittest.mock import patch
        from backend.infra.embedding import EmbeddingService

        old_state_path = os.environ.pop("BM25_STATE_PATH", None)
        try:
            with patch("backend.infra.embedding.MILVUS_COLLECTION", "collection_a"), \
                 patch("backend.infra.embedding.EVAL_RETRIEVAL_TEXT_MODE", "title_context"):
                svc_a = EmbeddingService()

            with patch("backend.infra.embedding.MILVUS_COLLECTION", "collection_b"), \
                 patch("backend.infra.embedding.EVAL_RETRIEVAL_TEXT_MODE", "title_context_filename"):
                svc_b = EmbeddingService()

            assert svc_a._state_path.name == "bm25_state_collection_a_title_context.json"
            assert svc_b._state_path.name == "bm25_state_collection_b_title_context_filename.json"
            assert svc_a._state_path != svc_b._state_path
        finally:
            if old_state_path is not None:
                os.environ["BM25_STATE_PATH"] = old_state_path

    def test_state_path_format(self):
        """Verify state path follows bm25_state_{collection}_{mode}.json pattern."""
        from backend.infra.embedding import _DEFAULT_STATE_PATH
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
            from backend.infra.embedding import EmbeddingService
            svc = EmbeddingService()
            # On Windows, Path normalizes slashes; compare Path objects
            assert str(svc._state_path).replace("\\", "/") == custom_path.replace("\\", "/")
        finally:
            if old is not None:
                os.environ["BM25_STATE_PATH"] = old
            else:
                os.environ.pop("BM25_STATE_PATH", None)

    def test_embedding_auto_device_falls_back_to_cpu(self):
        from unittest.mock import patch
        from backend.infra.embedding import _resolve_torch_device

        with patch("torch.cuda.is_available", return_value=False):
            assert _resolve_torch_device("auto") == "cpu"

    def test_embedding_gpu_only_requires_cuda(self):
        from unittest.mock import patch
        from backend.infra.embedding import _resolve_torch_device

        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="GPU-only"):
                _resolve_torch_device("cuda")
