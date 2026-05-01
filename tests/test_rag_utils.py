import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import backend.rag.utils as rag_utils  # noqa: E402
from backend.rag.runtime_config import load_layered_rerank_config, load_runtime_config  # noqa: E402


class RagUtilsDiagnosticsTests(unittest.TestCase):
    def test_runtime_config_loads_fake_env_without_reloading_modules(self):
        env = {
            "RAG_CANDIDATE_K": "77",
            "RERANK_TOP_N": "9",
            "MILVUS_SEARCH_EF": "111",
            "MILVUS_RRF_K": "44",
            "LAYERED_RERANK_ENABLED": "true",
            "L0_DENSE_TOP_K": "21",
            "L1_CHUNKS_PER_SCOPE_FILE": "8",
            "L2_CE_DEFAULT_K": "13",
            "L3_ROOT_WEIGHT": "0.27",
            "RAG_FAST_ENABLED": "true",
            "RAG_DEEP_ENABLED": "true",
            "RAG_FAST_EXPERIMENT": "shadow",
            "RAG_DEEP_TRACE": "shadow",
            "RAG_CITATION_VERIFY_ENABLED": "true",
            "RAG_UNIFIED_EXECUTION_ENABLED": "true",
            "RAG_DEEP_SHADOW": "true",
            "RAG_DEEP_ACTIVE": "true",
            "RAG_DEEP_MIN_COVERAGE": "0.5",
            "RAG_MODEL": "not-reserved",
            "RAG_INDEX_PROFILE": "fake-profile",
        }

        config = load_runtime_config(env)
        layered = load_layered_rerank_config(env)

        self.assertEqual(config.rag_candidate_k, 77)
        self.assertEqual(config.rerank_top_n, 9)
        self.assertEqual(config.milvus_search_ef, 111)
        self.assertEqual(config.milvus_rrf_k, 44)
        self.assertEqual(config.rag_index_profile, "fake-profile")
        self.assertEqual(config.execution_mode, "STANDARD")
        self.assertFalse(config.deep_executed)
        self.assertFalse(config.plan_applied)
        self.assertTrue(config.layered_candidate_enabled)
        self.assertTrue(config.citation_verify_enabled)
        self.assertTrue(config.unified_execution_enabled)
        self.assertTrue(config.deep_shadow_enabled)
        self.assertTrue(config.deep_active_enabled)
        self.assertEqual(config.deep_min_coverage, 0.5)
        self.assertEqual(config.reserved_flags["RAG_FAST_ENABLED"], "true")
        self.assertEqual(config.reserved_flags["RAG_DEEP_ENABLED"], "true")
        self.assertEqual(config.reserved_flags["RAG_FAST_EXPERIMENT"], "shadow")
        self.assertEqual(config.reserved_flags["RAG_DEEP_TRACE"], "shadow")
        self.assertNotIn("RAG_MODEL", config.reserved_flags)
        self.assertTrue(layered.enabled)
        self.assertEqual(layered.l0_dense_top_k, 21)
        self.assertEqual(layered.l1_chunks_per_scope_file, 8)
        self.assertEqual(layered.l2_ce_default_k, 13)
        self.assertEqual(layered.l3_root_weight, 0.27)

    def test_candidate_trace_includes_stable_signature_fields(self):
        traced = rag_utils._candidate_trace(
            [
                {
                    "chunk_id": "c1",
                    "filename": "manual.pdf",
                    "page_number": 2,
                    "retrieval_text": "heading\nbody",
                }
            ]
        )

        self.assertEqual(traced[0]["candidate_id"], "c1")
        self.assertTrue(traced[0]["text_hash"])

    def test_stage_error_uses_shared_type_shape(self):
        self.assertEqual(
            rag_utils._stage_error("rerank", "failed", "fallback"),
            {
                "stage": "rerank",
                "error": "failed",
                "fallback_to": "fallback",
                "severity": "warning",
                "recoverable": True,
                "user_visible": False,
            },
        )

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
            patch("backend.rag.utils._rerank_documents", side_effect=lambda query, docs, top_k: (docs[:top_k], {"rerank_enabled": False, "rerank_applied": False})),
            patch("backend.rag.utils._apply_structure_rerank", side_effect=lambda docs, top_k: (docs[:top_k], {"structure_rerank_enabled": False, "structure_rerank_applied": False})),
            patch("backend.rag.utils._evaluate_retrieval_confidence", return_value={"confidence_gate_enabled": False, "fallback_required": False, "confidence_reasons": []}),
        ):
            result = rag_utils.retrieve_documents("query", top_k=1)

        self.assertEqual(len(result["docs"]), 1)
        self.assertEqual(result["docs"][0]["chunk_id"], dense_doc["chunk_id"])
        self.assertEqual(result["meta"]["retrieval_mode"], "dense_fallback")
        self.assertIn("hybrid channel closed", result["meta"]["hybrid_error"])
        self.assertEqual(result["meta"]["candidate_count_after_rerank"], 1)

    def test_layered_failure_preserves_scoped_fallback_mode(self):
        trace_patch = {"scope_filter_applied": True}
        embeddings = rag_utils.QueryEmbeddings(dense=[0.1, 0.2], sparse={1: 0.5})
        docs = [{"text": "fallback", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.5}]

        with (
            patch.object(rag_utils._milvus_manager, "split_retrieve", side_effect=RuntimeError("split down")),
            patch.object(rag_utils._milvus_manager, "hybrid_retrieve", return_value=docs),
        ):
            result = rag_utils.retrieve_layered_candidates(
                embeddings,
                candidate_k=5,
                filter_expr='chunk_level == 3 and filename in ["manual.pdf"]',
                query_plan=rag_utils.QueryPlan(
                    raw_query="q",
                    semantic_query="q",
                    clean_query="q",
                    doc_hints=[],
                    matched_files=[("manual.pdf", 1.0)],
                    scope_mode="filter",
                    heading_hint=None,
                    anchors=[],
                    model_numbers=[],
                    intent_type=None,
                    route="scoped_hybrid",
                ),
                scope_matched_files=[("manual.pdf", 1.0)],
                timings={},
                trace_patch=trace_patch,
                retrieval_mode="hybrid_scoped",
            )

        self.assertEqual(result.retrieval_mode, "hybrid_scoped")
        self.assertEqual(result.candidates, docs)
        self.assertEqual(result.stage_errors[0]["stage"], "layered_retrieve")
        self.assertEqual(result.stage_errors[0]["fallback_to"], "standard_hybrid")

    def test_layered_sparse_failure_preserves_scoped_dense_mode(self):
        trace_patch = {"scope_filter_applied": True}
        embeddings = rag_utils.QueryEmbeddings(dense=[0.1, 0.2], sparse=None, sparse_error="sparse down")
        docs = [{"text": "fallback", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.5}]

        with patch.object(rag_utils._milvus_manager, "dense_retrieve", return_value=docs):
            result = rag_utils.retrieve_layered_candidates(
                embeddings,
                candidate_k=5,
                filter_expr='chunk_level == 3 and filename in ["manual.pdf"]',
                query_plan=rag_utils.QueryPlan(
                    raw_query="q",
                    semantic_query="q",
                    clean_query="q",
                    doc_hints=[],
                    matched_files=[("manual.pdf", 1.0)],
                    scope_mode="filter",
                    heading_hint=None,
                    anchors=[],
                    model_numbers=[],
                    intent_type=None,
                    route="scoped_hybrid",
                ),
                scope_matched_files=[("manual.pdf", 1.0)],
                timings={},
                trace_patch=trace_patch,
                retrieval_mode="hybrid_scoped",
            )

        self.assertEqual(result.retrieval_mode, "dense_fallback_scoped")
        self.assertEqual(result.candidates, docs)

    def test_layered_success_marks_candidate_strategy_not_rerank_strategy(self):
        trace_patch = {"query_plan_enabled": True}
        embeddings = rag_utils.QueryEmbeddings(dense=[0.1, 0.2], sparse={1: 0.5})
        docs = [
            {
                "text": "candidate",
                "retrieval_text": "candidate",
                "filename": "manual.pdf",
                "chunk_id": "c1",
                "score": 0.5,
                "dense_rank": 1,
                "sparse_rank": 1,
                "in_dense": True,
                "in_sparse": True,
            }
        ]

        with (
            patch.object(rag_utils._milvus_manager, "split_retrieve", return_value=docs),
            patch.object(rag_utils._milvus_manager, "hybrid_retrieve", return_value=[]),
        ):
            result = rag_utils.retrieve_layered_candidates(
                embeddings,
                candidate_k=5,
                filter_expr='chunk_level == 3',
                query_plan=rag_utils.QueryPlan(
                    raw_query="q",
                    semantic_query="q",
                    clean_query="q",
                    doc_hints=[],
                    matched_files=[],
                    scope_mode="none",
                    heading_hint=None,
                    anchors=[],
                    model_numbers=[],
                    intent_type=None,
                    route="global_hybrid",
                ),
                scope_matched_files=[],
                timings={},
                trace_patch=trace_patch,
                retrieval_mode="hybrid",
            )

        self.assertEqual(result.trace_patch["candidate_strategy"], "layered_split")
        self.assertEqual(result.trace_patch["rerank_strategy"], "shared_pipeline")

    def test_retrieval_knobs_are_passed_to_milvus_and_trace(self):
        docs = [
            {"text": "d1", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.9},
            {"text": "d2", "filename": "manual.pdf", "chunk_id": "c2", "score": 0.8},
        ]

        with (
            patch.object(rag_utils, "RAG_CANDIDATE_K", 80),
            patch.object(rag_utils, "RERANK_TOP_N", 20),
            patch.object(rag_utils, "RERANK_MODEL", ""),
            patch.object(rag_utils, "MILVUS_SEARCH_EF", 128),
            patch.object(rag_utils, "MILVUS_RRF_K", 70),
            patch.object(rag_utils, "MILVUS_SPARSE_DROP_RATIO", 0.1),
            patch.object(rag_utils._embedding_service, "get_embeddings", return_value=[[0.1, 0.2]]),
            patch.object(rag_utils._embedding_service, "get_sparse_embedding", return_value={1: 0.5}),
            patch.object(rag_utils._milvus_manager, "hybrid_retrieve", return_value=docs) as hybrid,
            patch("backend.rag.utils._apply_structure_rerank", side_effect=lambda docs, top_k: (docs[:top_k], {"structure_rerank_enabled": False, "structure_rerank_applied": False})),
            patch("backend.rag.utils._evaluate_retrieval_confidence", return_value={"confidence_gate_enabled": False, "fallback_required": False, "confidence_reasons": []}),
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

    def test_candidate_pool_retrieval_skips_rerank_pipeline(self):
        docs = [
            {"text": "d1", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.9},
            {"text": "d2", "filename": "manual.pdf", "chunk_id": "c2", "score": 0.8},
        ]

        with (
            patch.object(rag_utils._embedding_service, "get_embeddings", return_value=[[0.1, 0.2]]),
            patch.object(rag_utils._embedding_service, "get_sparse_embedding", return_value={1: 0.5}),
            patch.object(rag_utils._milvus_manager, "hybrid_retrieve", return_value=docs),
            patch("backend.rag.utils._rerank_documents", side_effect=AssertionError("rerank should not run")),
        ):
            result = rag_utils.retrieve_candidate_pool("query", top_k=1, candidate_k=3)

        self.assertEqual([doc["chunk_id"] for doc in result["candidates"]], ["c1", "c2"])
        self.assertTrue(result["meta"]["candidate_only"])
        self.assertFalse(result["meta"]["rerank_applied"])
        self.assertEqual(result["meta"]["candidate_k"], 3)

    def test_query_plan_disabled_does_not_load_filename_registry_or_change_query(self):
        docs = [{"text": "d1", "filename": "manual.pdf", "chunk_id": "c1", "score": 0.9}]

        with (
            patch.object(rag_utils, "QUERY_PLAN_ENABLED", False),
            patch.object(rag_utils._embedding_service, "get_embeddings", return_value=[[0.1, 0.2]]) as dense,
            patch.object(rag_utils._embedding_service, "get_sparse_embedding", return_value={1: 0.5}) as sparse,
            patch.object(rag_utils, "get_filename_registry", side_effect=AssertionError("registry should not load")),
            patch.object(rag_utils._milvus_manager, "hybrid_retrieve", return_value=docs),
            patch("backend.rag.utils._rerank_documents", side_effect=lambda query, docs, top_k: (docs[:top_k], {"rerank_enabled": False, "rerank_applied": False})),
            patch("backend.rag.utils._apply_structure_rerank", side_effect=lambda docs, top_k: (docs[:top_k], {"structure_rerank_enabled": False, "structure_rerank_applied": False})),
            patch("backend.rag.utils._evaluate_retrieval_confidence", return_value={"confidence_gate_enabled": False, "fallback_required": False, "confidence_reasons": []}),
        ):
            result = rag_utils.retrieve_documents("《Manual》中，如何配置？", top_k=1)

        dense.assert_called_with(["《Manual》中，如何配置？"])
        sparse.assert_called_with("《Manual》中，如何配置？")
        self.assertFalse(result["meta"]["query_plan_enabled"])
        self.assertEqual(result["meta"]["semantic_query"], "《Manual》中，如何配置？")

    def test_local_reranker_input_cap_limits_candidates(self):
        class FakeReranker:
            def predict(self, pairs):
                return [1.0 - i * 0.01 for i in range(len(pairs))]

        docs = [{"text": f"doc {idx}", "chunk_id": f"c{idx}", "score": 1.0 / (idx + 1)} for idx in range(50)]

        with (
            patch.object(rag_utils, "RERANK_PROVIDER", "local"),
            patch.object(rag_utils, "RERANK_MODEL", "fake-reranker"),
            patch.object(rag_utils, "RERANK_TOP_N", 0),
            patch.object(rag_utils, "RERANK_INPUT_K_GPU", 7),
            patch.object(rag_utils, "RERANK_DEVICE", "cuda"),
            patch.object(rag_utils, "RERANK_CACHE_ENABLED", False),
            patch("backend.rag.utils._resolve_rerank_device", return_value="cuda"),
            patch.object(rag_utils, "_get_local_reranker", return_value=FakeReranker()),
        ):
            reranked, meta = rag_utils._rerank_documents("query", docs, top_k=4)

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
            patch.object(rag_utils, "RERANK_TOP_N", 0),
            patch.object(rag_utils, "RERANK_INPUT_K_GPU", 10),
            patch.object(rag_utils, "RERANK_DEVICE", "cuda"),
            patch.object(rag_utils, "RERANK_CACHE_ENABLED", True),
            patch("backend.rag.utils._resolve_rerank_device", return_value="cuda"),
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

    def test_rerank_auto_device_uses_cpu_cap_when_cuda_unavailable(self):
        docs = [{"text": f"doc {idx}", "chunk_id": f"c{idx}", "score": 1.0} for idx in range(12)]

        with (
            patch.object(rag_utils, "RERANK_MODEL", "fake-reranker"),
            patch.object(rag_utils, "RERANK_INPUT_K_GPU", 10),
            patch.object(rag_utils, "RERANK_INPUT_K_CPU", 3),
            patch.object(rag_utils, "RERANK_DEVICE", "auto"),
            patch("backend.rag.utils._resolve_rerank_device", return_value="cpu"),
            patch.object(rag_utils, "RERANK_CACHE_ENABLED", False),
            patch.object(rag_utils, "_get_local_reranker", return_value=None),
        ):
            reranked, meta = rag_utils._rerank_documents("query", docs, top_k=5)

        self.assertEqual(meta["rerank_input_device_tier"], "cpu")
        self.assertEqual(meta["rerank_input_cap"], 3)
        self.assertEqual(meta["rerank_input_count"], 3)
        self.assertEqual(len(reranked), 3)

    def test_rerank_gpu_only_raises_when_cuda_missing(self):
        with (
            patch.object(rag_utils, "RERANK_DEVICE", "cuda"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU-only"):
                rag_utils._resolve_rerank_device()

    def test_rerank_dtype_resolves_rag_dtype_names(self):
        self.assertEqual(rag_utils.resolve_dtype("fp16"), "float16")
        self.assertEqual(rag_utils.resolve_dtype("bf16"), "bfloat16")
        self.assertEqual(rag_utils.resolve_dtype("fp32"), "float32")

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

        with patch.object(rag_utils, "SAME_ROOT_CAP", 2):
            reranked, meta = rag_utils._apply_structure_rerank(docs, top_k=5)

        self.assertEqual([doc["chunk_id"] for doc in reranked], ["a1", "b1", "a2", "c1"])
        self.assertAlmostEqual(meta["dominant_root_share"], 0.5825, places=3)
        self.assertEqual(meta["dominant_root_support"], 2)

    def test_rerank_score_fusion_can_preserve_strong_retrieval_rank(self):
        docs = [
            {"chunk_id": "c1", "rrf_rank": 1, "filename": "manual.pdf", "text": "best retrieval"},
            {"chunk_id": "c2", "rrf_rank": 20, "filename": "manual.pdf", "text": "best reranker"},
        ]

        with (
            patch.object(rag_utils, "RERANK_SCORE_FUSION_ENABLED", True),
            patch.object(rag_utils, "RERANK_FUSION_RERANK_WEIGHT", 0.0),
            patch.object(rag_utils, "RERANK_FUSION_RRF_WEIGHT", 1.0),
            patch.object(rag_utils, "RERANK_FUSION_SCOPE_WEIGHT", 0.0),
            patch.object(rag_utils, "RERANK_FUSION_METADATA_WEIGHT", 0.0),
            patch.object(rag_utils, "MILVUS_RRF_K", 60),
        ):
            fused = rag_utils._apply_rerank_score_fusion([(0, 0.1), (1, 0.9)], docs)

        self.assertEqual(fused[0][0], 0)
        self.assertGreater(fused[0][1], fused[1][1])

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
            patch.object(rag_utils, "_get_parent_chunk_store") as get_store,
        ):
            get_store.return_value.get_documents_by_ids.return_value = [parent]
            merged, meta = rag_utils._auto_merge_documents(docs, top_k=5)

        get_store.return_value.get_documents_by_ids.assert_called_once_with(["root-a"])
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

        with (
            patch.object(rag_utils, "CONFIDENCE_GATE_ENABLED", True),
            patch.object(rag_utils, "ENABLE_ANCHOR_GATE", True),
        ):
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
