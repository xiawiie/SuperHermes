import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import rag_pipeline  # noqa: E402
import rag_utils  # noqa: E402


class RagObservabilityTests(unittest.TestCase):
    def test_retrieve_documents_reports_stage_timings(self):
        docs = [{"text": "d1", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.9}]

        with (
            patch.object(rag_utils, "RERANK_PROVIDER", "api"),
            patch.object(rag_utils, "RERANK_MODEL", ""),
            patch.object(rag_utils._embedding_service, "get_embeddings", return_value=[[0.1, 0.2]]),
            patch.object(rag_utils._embedding_service, "get_sparse_embedding", return_value={1: 0.5}),
            patch.object(rag_utils._milvus_manager, "hybrid_retrieve", return_value=docs),
            patch("rag_utils._apply_structure_rerank", side_effect=lambda docs, top_k: (docs[:top_k], {})),
            patch("rag_utils._evaluate_retrieval_confidence", return_value={"fallback_required": False}),
        ):
            result = rag_utils.retrieve_documents("query", top_k=1)

        timings = result["meta"]["timings"]
        self.assertIn("embed_dense_ms", timings)
        self.assertIn("embed_sparse_ms", timings)
        self.assertIn("milvus_hybrid_ms", timings)
        self.assertIn("rerank_ms", timings)
        self.assertIn("structure_rerank_ms", timings)
        self.assertIn("confidence_ms", timings)
        self.assertIn("total_retrieve_ms", timings)
        self.assertEqual(result["meta"]["stage_errors"], [])

    def test_retrieve_documents_reports_fallback_stage_errors(self):
        docs = [{"text": "d1", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.9}]

        with (
            patch.object(rag_utils, "RERANK_PROVIDER", "api"),
            patch.object(rag_utils, "RERANK_MODEL", ""),
            patch.object(rag_utils._embedding_service, "get_embeddings", return_value=[[0.1, 0.2]]),
            patch.object(rag_utils._embedding_service, "get_sparse_embedding", return_value={1: 0.5}),
            patch.object(rag_utils._milvus_manager, "hybrid_retrieve", side_effect=RuntimeError("hybrid down")),
            patch.object(rag_utils._milvus_manager, "dense_retrieve", return_value=docs),
            patch("rag_utils._apply_structure_rerank", side_effect=lambda docs, top_k: (docs[:top_k], {})),
            patch("rag_utils._evaluate_retrieval_confidence", return_value={"fallback_required": False}),
        ):
            result = rag_utils.retrieve_documents("query", top_k=1)

        self.assertEqual(result["meta"]["retrieval_mode"], "dense_fallback")
        self.assertEqual(result["meta"]["stage_errors"][0]["stage"], "hybrid_retrieve")
        self.assertEqual(result["meta"]["stage_errors"][0]["fallback_to"], "dense_retrieve")
        self.assertIn("milvus_dense_fallback_ms", result["meta"]["timings"])

    def test_fast_path_grade_reports_grader_timing(self):
        state = {
            "question": "q",
            "context": "c",
            "rag_trace": {"fallback_required": False, "timings": {}},
        }

        with patch("rag_pipeline._get_grader_model", side_effect=AssertionError("grader should not be called")):
            result = rag_pipeline.grade_documents_node(state)

        self.assertEqual(result["route"], "generate_answer")
        self.assertIn("grader_ms", result["rag_trace"]["timings"])

    def test_retrieve_initial_propagates_rerank_cache_trace_fields(self):
        docs = [{"text": "d1", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.9}]
        meta = {
            "timings": {},
            "stage_errors": [],
            "rerank_enabled": True,
            "rerank_applied": True,
            "rerank_input_count": 20,
            "rerank_output_count": 5,
            "rerank_input_cap": 20,
            "rerank_input_device_tier": "gpu",
            "rerank_cache_enabled": True,
            "rerank_cache_hit": True,
            "fallback_required": False,
        }

        with patch("rag_pipeline.retrieve_documents", return_value={"docs": docs, "meta": meta}):
            result = rag_pipeline.retrieve_initial({"question": "query", "context_files": []})

        trace = result["rag_trace"]
        self.assertEqual(trace["rerank_input_count"], 20)
        self.assertEqual(trace["rerank_output_count"], 5)
        self.assertEqual(trace["rerank_input_cap"], 20)
        self.assertEqual(trace["rerank_input_device_tier"], "gpu")
        self.assertTrue(trace["rerank_cache_enabled"])
        self.assertTrue(trace["rerank_cache_hit"])


if __name__ == "__main__":
    unittest.main()
