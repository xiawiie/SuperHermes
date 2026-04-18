import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import rag_utils  # noqa: E402


class RagUtilsDiagnosticsTests(unittest.TestCase):
    def test_dense_fallback_keeps_hybrid_error_in_meta(self):
        dense_doc = {"text": "fallback text", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.5}

        with (
            patch.object(rag_utils._embedding_service, "get_embeddings", return_value=[[0.1, 0.2]]),
            patch.object(rag_utils._embedding_service, "get_sparse_embedding", return_value={1: 0.5}),
            patch.object(rag_utils._milvus_manager, "hybrid_retrieve", side_effect=RuntimeError("hybrid channel closed")),
            patch.object(rag_utils._milvus_manager, "dense_retrieve", return_value=[dense_doc]),
            patch("rag_utils._rerank_documents", side_effect=lambda query, docs, top_k: (docs[:top_k], {"rerank_enabled": False, "rerank_applied": False})),
            patch("rag_utils._auto_merge_documents", side_effect=lambda docs, top_k: (docs[:top_k], {"auto_merge_enabled": True, "auto_merge_applied": False})),
        ):
            result = rag_utils.retrieve_documents("query", top_k=1)

        self.assertEqual(result["docs"], [dense_doc])
        self.assertEqual(result["meta"]["retrieval_mode"], "dense_fallback")
        self.assertIn("hybrid channel closed", result["meta"]["hybrid_error"])

    def test_total_retrieval_failure_reports_hybrid_and_dense_errors(self):
        with (
            patch.object(rag_utils._embedding_service, "get_embeddings", return_value=[[0.1, 0.2]]),
            patch.object(rag_utils._embedding_service, "get_sparse_embedding", return_value={1: 0.5}),
            patch.object(rag_utils._milvus_manager, "hybrid_retrieve", side_effect=RuntimeError("hybrid down")),
            patch.object(rag_utils._milvus_manager, "dense_retrieve", side_effect=RuntimeError("dense down")),
        ):
            result = rag_utils.retrieve_documents("query", top_k=1)

        self.assertEqual(result["docs"], [])
        self.assertEqual(result["meta"]["retrieval_mode"], "failed")
        self.assertIn("hybrid down", result["meta"]["hybrid_error"])
        self.assertIn("dense down", result["meta"]["dense_error"])


if __name__ == "__main__":
    unittest.main()
