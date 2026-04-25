import unittest
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_rag_matrix import (
    VARIANT_CONFIGS,
    compare_sample_rank,
    compute_retrieval_metrics,
    evaluate_sample,
    first_relevant_rank,
    parse_variants,
    render_summary_markdown,
    summarize_results,
    validate_eval_dataset_records,
    validate_variant_order,
)


class EvaluateRagMatrixMetricTests(unittest.TestCase):
    def test_compute_metrics_uses_root_anchor_keyword_matches(self):
        docs = [
            {"chunk_id": "leaf-a", "root_chunk_id": "wrong", "text": "noise"},
            {
                "chunk_id": "leaf-b",
                "root_chunk_id": "root-1",
                "section_title": "第二条",
                "text": "自然人 民事关系",
            },
        ]

        metrics = compute_retrieval_metrics(
            docs,
            expected_chunk_ids=[],
            expected_root_ids=["root-1"],
            expected_anchors=["第二条"],
            expected_keywords=["自然人"],
            top_k=5,
        )

        self.assertTrue(metrics["hit_at_5"])
        self.assertTrue(metrics["root_hit_at_5"])
        self.assertTrue(metrics["anchor_hit_at_5"])
        self.assertTrue(metrics["keyword_hit_at_5"])
        self.assertEqual(metrics["first_relevant_rank"], 2)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["context_precision_id_at_5"], 0.5)
        self.assertEqual(metrics["irrelevant_context_ratio_at_5"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_5"], 1.0)

    def test_recall_counts_distinct_expected_items_found(self):
        docs = [
            {"chunk_id": "leaf-a", "root_chunk_id": "root-1", "text": "自然人"},
            {"chunk_id": "leaf-b", "root_chunk_id": "wrong", "text": "noise"},
        ]
        metrics = compute_retrieval_metrics(
            docs,
            expected_root_ids=["root-1", "root-2"],
            expected_keywords=["自然人", "法人"],
            top_k=5,
        )
        # root-1 found, root-2 not found = 1/2; "自然人" found, "法人" not = 1/2; total 2/4
        self.assertAlmostEqual(metrics["recall_at_5"], 0.5)

    def test_recall_is_none_when_no_expected(self):
        docs = [{"chunk_id": "leaf-a", "text": "whatever"}]
        metrics = compute_retrieval_metrics(docs, top_k=5)
        self.assertIsNone(metrics["recall_at_5"])

    def test_id_metrics_use_positive_context_qrels(self):
        docs = [
            {"filename": "manual.pdf", "page_number": 4, "chunk_id": "wrong"},
            {"filename": "other.pdf", "page_number": 1, "chunk_id": "noise"},
        ]

        metrics = compute_retrieval_metrics(
            docs,
            positive_contexts=[{"doc_id": "manual.pdf::p4", "relevance": 3}],
            relevance_judgments=[{"doc_id": "manual.pdf::p4", "relevance": 3}],
            hard_negative_files=["other.pdf"],
            top_k=5,
        )

        self.assertEqual(metrics["id_context_recall_at_5"], 1.0)
        self.assertEqual(metrics["id_context_precision_at_5"], 0.5)
        self.assertGreater(metrics["ndcg_at_5"], 0.0)
        self.assertTrue(metrics["hard_negative_file_hit_at_5"])
        self.assertEqual(metrics["hard_negative_context_ratio_at_5"], 0.5)

    def test_validate_eval_dataset_records_requires_benchmark_fields(self):
        report = validate_eval_dataset_records(
            [
                {
                    "id": "r1",
                    "query": "q",
                    "reference_answer": "a",
                    "expected_files": ["manual.pdf"],
                    "expected_pages": [1],
                    "positive_contexts": [{"doc_id": "manual.pdf::p1"}],
                    "relevance_judgments": [{"doc_id": "manual.pdf::p1", "relevance": 3}],
                    "hard_negative_files": ["other.pdf"],
                    "expected_keyword_policy": {"mode": "at_least", "min_match": 1, "total": 3},
                }
            ],
            dataset=Path("dataset.jsonl"),
        )

        self.assertTrue(report["ok"])

        bad_report = validate_eval_dataset_records([{"id": "bad"}], dataset=Path("bad.jsonl"))
        self.assertFalse(bad_report["ok"])

    def test_first_relevant_rank_uses_legacy_chunk_id(self):
        docs = [
            {"chunk_id": "leaf-a", "root_chunk_id": "wrong", "text": "noise"},
            {"chunk_id": "legacy-1", "root_chunk_id": "wrong", "text": "more noise"},
        ]

        rank = first_relevant_rank(docs, expected_chunk_ids=["legacy-1"], top_k=5)

        self.assertEqual(rank, 2)

    def test_anchor_match_avoids_partial_numeric_anchor_hits(self):
        docs = [{"chunk_id": "leaf-a", "root_chunk_id": "wrong", "section_title": "11.2 高级配置"}]

        metrics = compute_retrieval_metrics(docs, expected_anchors=["1.2"], top_k=5)

        self.assertFalse(metrics["hit_at_5"])
        self.assertFalse(metrics["anchor_hit_at_5"])

    def test_anchor_match_correctly_matches_legal_article(self):
        docs = [{"chunk_id": "leaf-a", "root_chunk_id": "root-1", "section_title": "第一条", "text": "为了保护民事主体的合法权益..."}]

        metrics = compute_retrieval_metrics(docs, expected_anchors=["第一条"], top_k=5)

        self.assertTrue(metrics["anchor_hit_at_5"])

    def test_anchor_match_no_false_positive_from_adjacent_numeral(self):
        docs = [{"chunk_id": "leaf-a", "root_chunk_id": "wrong", "text": "二十一、其他事项"}]

        metrics = compute_retrieval_metrics(docs, expected_anchors=["一、"], top_k=5)

        self.assertFalse(metrics["anchor_hit_at_5"])

    def test_chunk_hit_takes_precedence_when_gold_chunk_ids_exist(self):
        docs = [
            {"chunk_id": "leaf-a", "root_chunk_id": "root-wrong", "text": "keyword"},
            {"chunk_id": "leaf-b", "root_chunk_id": "root-1", "text": "keyword"},
        ]

        metrics = compute_retrieval_metrics(
            docs,
            expected_chunk_ids=["leaf-b"],
            expected_root_ids=["root-1"],
            expected_keywords=["keyword"],
            top_k=5,
        )

        self.assertTrue(metrics["chunk_hit_at_5"])
        self.assertTrue(metrics["hit_at_5"])
        self.assertEqual(metrics["first_relevant_rank"], 2)

    def test_compare_sample_rank_counts_win_loss_tie(self):
        old = {"hit_at_5": False, "first_relevant_rank": None}
        new = {"hit_at_5": True, "first_relevant_rank": 3}

        self.assertEqual(compare_sample_rank(old, new), "win")
        self.assertEqual(compare_sample_rank(new, old), "loss")
        self.assertEqual(compare_sample_rank(new, {"hit_at_5": True, "first_relevant_rank": 3}), "tie")

    def test_summarize_results_counts_diagnostics_and_pairs(self):
        rows = [
            {
                "sample_id": "s1",
                "variant": "A0",
                "metrics": {
                    "hit_at_5": False,
                    "first_relevant_rank": None,
                    "mrr": 0.0,
                    "error_rate": 0.0,
                    "fallback_helped": False,
                    "fallback_hurt": False,
                },
                "diagnostic_result": {"category": "recall_miss"},
                "latency_ms": 100.0,
                "fallback_required": False,
                "rewrite_strategy": "none",
            },
            {
                "sample_id": "s1",
                "variant": "A1",
                "metrics": {
                    "hit_at_5": True,
                    "first_relevant_rank": 2,
                    "mrr": 0.5,
                    "error_rate": 0.0,
                    "fallback_helped": True,
                    "fallback_hurt": False,
                },
                "diagnostic_result": {"category": "ok"},
                "latency_ms": 200.0,
                "fallback_required": True,
                "rewrite_strategy": "step_back",
            },
            {
                "sample_id": "s2",
                "variant": "A0",
                "metrics": {
                    "hit_at_5": True,
                    "first_relevant_rank": 1,
                    "mrr": 1.0,
                    "error_rate": 0.0,
                    "fallback_helped": False,
                    "fallback_hurt": False,
                },
                "diagnostic_result": {"category": "ok"},
                "latency_ms": 100.0,
                "fallback_required": False,
                "rewrite_strategy": "none",
            },
            {
                "sample_id": "s2",
                "variant": "A1",
                "metrics": {
                    "hit_at_5": True,
                    "first_relevant_rank": 3,
                    "mrr": 0.333,
                    "error_rate": 0.0,
                    "fallback_helped": False,
                    "fallback_hurt": False,
                },
                "diagnostic_result": {"category": "ranking_miss"},
                "latency_ms": 200.0,
                "fallback_required": False,
                "rewrite_strategy": "none",
            },
        ]

        summary = summarize_results(rows, variants=["A0", "A1"])

        self.assertEqual(summary["paired_comparisons"]["A1_vs_A0"]["wins"], 1)
        self.assertEqual(summary["paired_comparisons"]["A1_vs_A0"]["losses"], 1)
        self.assertEqual(summary["variants"]["A1"]["hit_at_5"], 1.0)
        self.assertEqual(summary["variants"]["A1"]["avg_latency_ms"], 200.0)
        self.assertEqual(summary["variants"]["A1"]["fallback_trigger_rate"], 0.5)
        self.assertEqual(summary["variants"]["A1"]["fallback_helped_rate"], 0.5)
        self.assertEqual(summary["variants"]["A1"]["rewrite_strategy_distribution"], {"step_back": 1, "none": 1})
        self.assertEqual(summary["diagnostics"]["A0"]["recall_miss"], 1)

    def test_phase2_sparse_variants_are_available(self):
        self.assertEqual(parse_variants("P1,P2,P3"), ["P1", "P2", "P3"])
        self.assertEqual(VARIANT_CONFIGS["P1"]["env"]["RERANK_TOP_N"], "0")
        self.assertEqual(VARIANT_CONFIGS["P1"]["env"]["MILVUS_SPARSE_DROP_RATIO"], "0.1")
        self.assertEqual(VARIANT_CONFIGS["P2"]["env"]["RERANK_TOP_N"], "0")
        self.assertEqual(VARIANT_CONFIGS["P2"]["env"]["MILVUS_RRF_K"], "100")
        self.assertEqual(VARIANT_CONFIGS["P3"]["env"]["RERANK_TOP_N"], "0")
        self.assertEqual(VARIANT_CONFIGS["P3"]["env"]["MILVUS_SPARSE_DROP_RATIO"], "0.1")
        self.assertEqual(VARIANT_CONFIGS["P3"]["env"]["MILVUS_RRF_K"], "100")

    def test_phase25_fallback_variant_is_b0_with_confidence_gate(self):
        self.assertEqual(parse_variants("B0,F1"), ["B0", "F1"])
        self.assertEqual(VARIANT_CONFIGS["F1"]["env"]["RERANK_TOP_N"], "0")
        self.assertEqual(VARIANT_CONFIGS["F1"]["env"]["STRUCTURE_RERANK_ENABLED"], "false")
        self.assertEqual(VARIANT_CONFIGS["F1"]["env"]["CONFIDENCE_GATE_ENABLED"], "true")
        self.assertEqual(VARIANT_CONFIGS["F1"]["env"]["ENABLE_ANCHOR_GATE"], "true")

    def test_v4_final_variant_names_are_available(self):
        self.assertEqual(
            parse_variants("B0_legacy,S1_linear,S2,S2H,S2HR,S3"),
            ["B0_legacy", "S1_linear", "S2", "S2H", "S2HR", "S3"],
        )
        self.assertEqual(VARIANT_CONFIGS["B0_legacy"]["env"], {})
        self.assertEqual(VARIANT_CONFIGS["S1_linear"]["env"]["CONFIDENCE_GATE_ENABLED"], "false")
        self.assertEqual(VARIANT_CONFIGS["S1_linear"]["env"]["RAG_FALLBACK_ENABLED"], "false")
        self.assertEqual(VARIANT_CONFIGS["S2"]["env"]["DOC_SCOPE_GLOBAL_RESERVE_WEIGHT"], "0.2")
        self.assertEqual(VARIANT_CONFIGS["S2HR"]["env"]["RERANK_PAIR_ENRICHMENT_ENABLED"], "true")

    def test_chunk_root_qrels_na_rendered_in_all_summary_sections(self):
        rows = [
            {
                "sample_id": "s1",
                "variant": "B0_legacy",
                "metrics": {"hit_at_5": False, "mrr": 0.0, "file_hit_at_5": False},
                "diagnostic_result": {"category": "file_recall_miss"},
                "latency_ms": 100.0,
            },
            {
                "sample_id": "s1",
                "variant": "S1_linear",
                "metrics": {"hit_at_5": False, "mrr": 0.0, "file_hit_at_5": False},
                "diagnostic_result": {"category": "ranking_miss"},
                "latency_ms": 90.0,
            },
        ]

        summary = summarize_results(rows, variants=["B0_legacy", "S1_linear"])
        rendered = render_summary_markdown(summary)

        self.assertIn("| B0_legacy |", rendered)
        self.assertIn("Chunk@5=n/a (qrels missing)", rendered)
        self.assertIn("Root@5=n/a (qrels missing)", rendered)
        self.assertIn("S1_linear_vs_B0_legacy", rendered)

    def test_evaluate_sample_graph_mode_records_fallback_helped(self):
        final_doc = {"filename": "manual.pdf", "page_number": 1, "text": "answer"}
        initial_doc = {"filename": "other.pdf", "page_number": 9, "text": "miss"}

        def classify_failure(**kwargs):
            return {"category": "ok", "failed_stage": None, "suggestions": []}

        def run_graph(question, context_files=None):
            return {
                "docs": [final_doc],
                "rag_trace": {
                    "retrieved_chunks": [final_doc],
                    "initial_retrieved_chunks": [initial_doc],
                    "fallback_required": True,
                    "rewrite_needed": True,
                    "retrieval_stage": "expanded",
                    "rewrite_strategy": "step_back",
                },
            }

        with patch(
            "scripts.evaluate_rag_matrix._ensure_backend_imports",
            return_value=(classify_failure, None, run_graph),
        ):
            row = evaluate_sample(
                {
                    "id": "s1",
                    "query": "q",
                    "expected_files": ["manual.pdf"],
                    "expected_pages": [1],
                    "positive_contexts": [{"doc_id": "manual.pdf::p1", "relevance": 3}],
                    "relevance_judgments": [{"doc_id": "manual.pdf::p1", "relevance": 3}],
                    "hard_negative_files": ["other.pdf"],
                    "expected_keyword_policy": {"mode": "at_least", "min_match": 1, "total": 1},
                },
                variant="B0",
                top_k=5,
                mode="graph",
            )

        self.assertTrue(row["metrics"]["final_retrieval_hit_at_5"])
        self.assertFalse(row["metrics"]["initial_retrieval_hit_at_5"])
        self.assertTrue(row["fallback_helped"])
        self.assertEqual(row["rewrite_strategy"], "step_back")

    def test_validate_variant_order_requires_title_index_for_b_group(self):
        with self.assertRaisesRegex(RuntimeError, "expects title_context"):
            validate_variant_order(["B1"], skip_reindex=False)

        validate_variant_order(["B1"], skip_reindex=True)
        validate_variant_order(["A1", "B1", "G1"], skip_reindex=False)


if __name__ == "__main__":
    unittest.main()
