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

        self.assertEqual(len(result["docs"]), 1)
        self.assertEqual(result["docs"][0]["chunk_id"], dense_doc["chunk_id"])
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

    def test_structure_rerank_boosts_leafs_from_strong_root_and_limits_duplicates(self):
        docs = [
            {"chunk_id": "a1", "root_chunk_id": "root-a", "filename": "manual.pdf", "text": "a1", "rerank_score": 0.82},
            {"chunk_id": "a2", "root_chunk_id": "root-a", "filename": "manual.pdf", "text": "a2", "rerank_score": 0.65},
            {"chunk_id": "a3", "root_chunk_id": "root-a", "filename": "manual.pdf", "text": "a3", "rerank_score": 0.61},
            {"chunk_id": "b1", "root_chunk_id": "root-b", "filename": "manual.pdf", "text": "b1", "rerank_score": 0.79},
            {"chunk_id": "c1", "root_chunk_id": "root-c", "filename": "manual.pdf", "text": "c1", "rerank_score": 0.3},
        ]

        reranked, meta = rag_utils._apply_structure_rerank(docs, top_k=5)

        self.assertEqual([doc["chunk_id"] for doc in reranked], ["a1", "b1", "a2", "c1"])
        self.assertAlmostEqual(meta["dominant_root_share"], 0.5825, places=3)
        self.assertEqual(meta["dominant_root_support"], 2)

    def test_structure_rerank_can_be_disabled_for_experiments(self):
        docs = [
            {"chunk_id": "a1", "root_chunk_id": "root-a", "filename": "manual.pdf", "text": "a1", "rerank_score": 0.5},
            {"chunk_id": "b1", "root_chunk_id": "root-b", "filename": "manual.pdf", "text": "b1", "rerank_score": 0.4},
        ]

        with patch.object(rag_utils, "STRUCTURE_RERANK_ENABLED", False):
            reranked, meta = rag_utils._apply_structure_rerank(docs, top_k=1)

        self.assertEqual([doc["chunk_id"] for doc in reranked], ["a1"])
        self.assertFalse(meta["structure_rerank_enabled"])
        self.assertFalse(meta["structure_rerank_applied"])

    def test_confidence_gate_requests_fallback_on_anchor_mismatch(self):
        docs = [
            {
                "chunk_id": "x1",
                "root_chunk_id": "root-x",
                "filename": "law.pdf",
                "text": "第一章总则的内容。",
                "retrieval_text": "总则\n第一章总则的内容。",
                "section_title": "总则",
                "anchor_id": "第一章",
                "final_score": 0.92,
            },
            {
                "chunk_id": "x2",
                "root_chunk_id": "root-x",
                "filename": "law.pdf",
                "text": "总则说明。",
                "retrieval_text": "总则\n总则说明。",
                "section_title": "总则",
                "anchor_id": "第一章",
                "final_score": 0.87,
            },
        ]

        meta = rag_utils._evaluate_retrieval_confidence("《中华人民共和国民法典》第二条主要规定了什么？", docs)

        self.assertTrue(meta["fallback_required"])
        self.assertIn("anchor_mismatch", meta["confidence_reasons"])

    def test_confidence_gate_can_be_disabled_for_experiments(self):
        docs = [
            {
                "chunk_id": "x1",
                "root_chunk_id": "root-x",
                "filename": "law.pdf",
                "text": "第一章总则的内容。",
                "retrieval_text": "总则\n第一章总则的内容。",
                "section_title": "总则",
                "anchor_id": "第一章",
                "final_score": 0.2,
            }
        ]

        with patch.object(rag_utils, "CONFIDENCE_GATE_ENABLED", False):
            meta = rag_utils._evaluate_retrieval_confidence("第二条是什么？", docs)

        self.assertFalse(meta["fallback_required"])
        self.assertEqual(meta["confidence_reasons"], [])


if __name__ == "__main__":
    unittest.main()
