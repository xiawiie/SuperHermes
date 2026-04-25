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
            patch.object(rag_utils, "RAG_CANDIDATE_K", 0),
            patch.object(rag_utils, "MILVUS_SEARCH_EF", 64),
            patch.object(rag_utils, "MILVUS_RRF_K", 60),
            patch.object(rag_utils, "MILVUS_SPARSE_DROP_RATIO", 0.2),
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
        self.assertEqual(result["meta"]["candidate_count_after_rerank"], 1)

    def test_retrieval_knobs_are_passed_to_milvus_and_trace(self):
        docs = [
            {"text": "d1", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.9},
            {"text": "d2", "filename": "manual.pdf", "chunk_id": "c2", "score": 0.8},
        ]

        with (
            patch.object(rag_utils, "RAG_CANDIDATE_K", 80),
            patch.object(rag_utils, "RERANK_TOP_N", 20),
            patch.object(rag_utils, "RERANK_PROVIDER", "api"),
            patch.object(rag_utils, "RERANK_MODEL", ""),
            patch.object(rag_utils, "RERANK_API_KEY", ""),
            patch.object(rag_utils, "RERANK_BINDING_HOST", ""),
            patch.object(rag_utils, "MILVUS_SEARCH_EF", 128),
            patch.object(rag_utils, "MILVUS_RRF_K", 70),
            patch.object(rag_utils, "MILVUS_SPARSE_DROP_RATIO", 0.1),
            patch.object(rag_utils._embedding_service, "get_embeddings", return_value=[[0.1, 0.2]]),
            patch.object(rag_utils._embedding_service, "get_sparse_embedding", return_value={1: 0.5}),
            patch.object(rag_utils._milvus_manager, "hybrid_retrieve", return_value=docs) as hybrid,
            patch("rag_utils._apply_structure_rerank", side_effect=lambda docs, top_k: (docs[:top_k], {"structure_rerank_enabled": False, "structure_rerank_applied": False})),
            patch("rag_utils._evaluate_retrieval_confidence", return_value={"confidence_gate_enabled": False, "fallback_required": False, "confidence_reasons": []}),
        ):
            result = rag_utils.retrieve_documents("query", top_k=1)

        hybrid.assert_called_once()
        kwargs = hybrid.call_args.kwargs
        self.assertEqual(kwargs["top_k"], 80)
        self.assertEqual(kwargs["search_ef"], 128)
        self.assertEqual(kwargs["rrf_k"], 70)
        self.assertEqual(kwargs["sparse_drop_ratio"], 0.1)
        self.assertEqual(result["meta"]["rerank_top_n"], 2)
        self.assertEqual(result["meta"]["candidate_count_before_rerank"], 2)
        self.assertEqual(result["meta"]["milvus_search_ef"], 128)

    def test_cpu_rerank_top_n_cap_limits_deep_variants(self):
        with (
            patch.object(rag_utils, "RERANK_DEVICE", "cpu"),
            patch.object(rag_utils, "RERANK_TOP_N", 30),
            patch.object(rag_utils, "RERANK_CPU_TOP_N_CAP", 10),
        ):
            self.assertEqual(rag_utils._effective_rerank_top_n(top_k=5, candidate_count=80), 10)

    def test_local_reranker_input_cap_limits_predict_pairs(self):
        class FakeReranker:
            def __init__(self):
                self.pair_count = None

            def predict(self, pairs):
                self.pair_count = len(pairs)
                return list(range(len(pairs)))

        docs = [{"text": f"doc {idx}", "chunk_id": f"c{idx}", "score": 1.0 / (idx + 1)} for idx in range(50)]
        fake_reranker = FakeReranker()

        with (
            patch.object(rag_utils, "RERANK_PROVIDER", "local"),
            patch.object(rag_utils, "RERANK_MODEL", "fake-reranker"),
            patch.object(rag_utils, "RERANK_DEVICE", "cpu"),
            patch.object(rag_utils, "RERANK_TOP_N", 0),
            patch.object(rag_utils, "RERANK_CPU_TOP_N_CAP", 0),
            patch.object(rag_utils, "RERANK_INPUT_K_CPU", 10),
            patch.object(rag_utils, "_get_local_reranker", return_value=fake_reranker),
        ):
            reranked, meta = rag_utils._rerank_documents("query", docs, top_k=5)

        self.assertEqual(fake_reranker.pair_count, 10)
        self.assertEqual(meta["candidate_count"], 50)
        self.assertEqual(meta["rerank_input_count"], 10)
        self.assertEqual(meta["rerank_output_count"], 5)
        self.assertEqual(len(reranked), 5)

    def test_api_reranker_input_cap_limits_payload_documents(self):
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"results": [{"index": idx, "relevance_score": 1.0 - idx * 0.01} for idx in range(4)]}

        captured_payloads = []

        def fake_post(*args, **kwargs):
            captured_payloads.append(kwargs["json"])
            return FakeResponse()

        docs = [{"text": f"doc {idx}", "chunk_id": f"c{idx}", "score": 1.0 / (idx + 1)} for idx in range(50)]

        with (
            patch.object(rag_utils, "RERANK_PROVIDER", "api"),
            patch.object(rag_utils, "RERANK_MODEL", "fake-reranker"),
            patch.object(rag_utils, "RERANK_API_KEY", "key"),
            patch.object(rag_utils, "RERANK_BINDING_HOST", "http://reranker"),
            patch.object(rag_utils, "RERANK_DEVICE", "cuda"),
            patch.object(rag_utils, "RERANK_TOP_N", 0),
            patch.object(rag_utils, "RERANK_INPUT_K_GPU", 7),
            patch("rag_utils.requests.post", side_effect=fake_post),
        ):
            reranked, meta = rag_utils._rerank_documents("query", docs, top_k=4)

        self.assertEqual(len(captured_payloads[0]["documents"]), 7)
        self.assertEqual(captured_payloads[0]["top_n"], 4)
        self.assertEqual(meta["rerank_input_count"], 7)
        self.assertEqual(meta["rerank_output_count"], 4)
        self.assertEqual(len(reranked), 4)

    def test_reranker_cache_hit_skips_local_predict(self):
        class FakeCache:
            def __init__(self):
                self.store = {}

            def get_string(self, key):
                if key == "milvus_index_version":
                    return "7"
                return None

            def get_json(self, key):
                return self.store.get(key)

            def set_json(self, key, value, ttl=None):
                self.store[key] = value

        class FakeReranker:
            def __init__(self):
                self.calls = 0

            def predict(self, pairs):
                self.calls += 1
                return list(range(len(pairs)))

        docs = [{"text": f"doc {idx}", "chunk_id": f"c{idx}", "score": 1.0 / (idx + 1)} for idx in range(12)]
        fake_cache = FakeCache()
        fake_reranker = FakeReranker()

        with (
            patch.object(rag_utils, "RERANK_PROVIDER", "local"),
            patch.object(rag_utils, "RERANK_MODEL", "fake-reranker"),
            patch.object(rag_utils, "RERANK_DEVICE", "cpu"),
            patch.object(rag_utils, "RERANK_TOP_N", 0),
            patch.object(rag_utils, "RERANK_INPUT_K_CPU", 10),
            patch.object(rag_utils, "RERANK_CACHE_ENABLED", True),
            patch.object(rag_utils, "cache", fake_cache),
            patch.object(rag_utils._embedding_service, "_total_docs", 42),
            patch.object(rag_utils, "_get_local_reranker", return_value=fake_reranker),
        ):
            first, first_meta = rag_utils._rerank_documents("query", docs, top_k=5)
            second, second_meta = rag_utils._rerank_documents("query", docs, top_k=5)

        self.assertEqual(fake_reranker.calls, 1)
        self.assertFalse(first_meta["rerank_cache_hit"])
        self.assertTrue(second_meta["rerank_cache_hit"])
        self.assertEqual([doc["chunk_id"] for doc in first], [doc["chunk_id"] for doc in second])

    def test_step_back_expand_uses_single_structured_llm_call(self):
        class FakeResponse:
            content = '{"step_back_question":"通用问题是什么？","step_back_answer":"通用答案。"}'

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def invoke(self, prompt):
                self.calls += 1
                return FakeResponse()

        fake_model = FakeModel()

        with patch.object(rag_utils, "_get_stepback_model", return_value=fake_model):
            result = rag_utils.step_back_expand("具体问题？")

        self.assertEqual(fake_model.calls, 1)
        self.assertEqual(result["step_back_question"], "通用问题是什么？")
        self.assertEqual(result["step_back_answer"], "通用答案。")
        self.assertIn("通用问题是什么？", result["expanded_query"])

    def test_step_back_expand_falls_back_on_invalid_json(self):
        class FakeResponse:
            content = "not json"

        class FakeModel:
            def __init__(self):
                self.calls = 0

            def invoke(self, prompt):
                self.calls += 1
                return FakeResponse()

        fake_model = FakeModel()

        with patch.object(rag_utils, "_get_stepback_model", return_value=fake_model):
            result = rag_utils.step_back_expand("具体问题？")

        self.assertEqual(fake_model.calls, 1)
        self.assertEqual(result["step_back_question"], "")
        self.assertEqual(result["step_back_answer"], "")
        self.assertEqual(result["expanded_query"], "具体问题？")

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

    def test_auto_merge_uses_single_existing_l3_to_l1_path(self):
        docs = [
            {"chunk_id": "leaf-a1", "parent_chunk_id": "root-a", "filename": "manual.pdf", "text": "a1", "score": 0.8},
            {"chunk_id": "leaf-a2", "parent_chunk_id": "root-a", "filename": "manual.pdf", "text": "a2", "score": 0.7},
            {"chunk_id": "leaf-b1", "parent_chunk_id": "root-b", "filename": "manual.pdf", "text": "b1", "score": 0.6},
        ]
        parent = {
            "chunk_id": "root-a",
            "parent_chunk_id": "",
            "root_chunk_id": "root-a",
            "filename": "manual.pdf",
            "text": "root a",
            "score": 0.0,
        }

        with (
            patch.object(rag_utils, "AUTO_MERGE_ENABLED", True),
            patch.object(rag_utils, "AUTO_MERGE_THRESHOLD", 2),
            patch.object(rag_utils._parent_chunk_store, "get_documents_by_ids", return_value=[parent]) as get_many,
        ):
            merged, meta = rag_utils._auto_merge_documents(docs, top_k=5)

        get_many.assert_called_once_with(["root-a"])
        self.assertEqual([doc["chunk_id"] for doc in merged], ["root-a", "leaf-b1"])
        self.assertTrue(meta["auto_merge_applied"])
        self.assertEqual(meta["auto_merge_steps"], 1)
        self.assertEqual(meta["auto_merge_path"], "L3->L1")

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

        with patch.object(rag_utils, "CONFIDENCE_GATE_ENABLED", True):
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
