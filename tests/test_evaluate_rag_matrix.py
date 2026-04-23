import unittest

from scripts.evaluate_rag_matrix import (
    compare_sample_rank,
    compute_retrieval_metrics,
    first_relevant_rank,
    summarize_results,
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
                "metrics": {"hit_at_5": False, "first_relevant_rank": None, "mrr": 0.0, "error_rate": 0.0},
                "diagnostic_result": {"category": "recall_miss"},
                "latency_ms": 100.0,
                "fallback_required": False,
            },
            {
                "sample_id": "s1",
                "variant": "A1",
                "metrics": {"hit_at_5": True, "first_relevant_rank": 2, "mrr": 0.5, "error_rate": 0.0},
                "diagnostic_result": {"category": "ok"},
                "latency_ms": 200.0,
                "fallback_required": True,
            },
            {
                "sample_id": "s2",
                "variant": "A0",
                "metrics": {"hit_at_5": True, "first_relevant_rank": 1, "mrr": 1.0, "error_rate": 0.0},
                "diagnostic_result": {"category": "ok"},
                "latency_ms": 100.0,
                "fallback_required": False,
            },
            {
                "sample_id": "s2",
                "variant": "A1",
                "metrics": {"hit_at_5": True, "first_relevant_rank": 3, "mrr": 0.333, "error_rate": 0.0},
                "diagnostic_result": {"category": "ranking_miss"},
                "latency_ms": 200.0,
                "fallback_required": False,
            },
        ]

        summary = summarize_results(rows, variants=["A0", "A1"])

        self.assertEqual(summary["paired_comparisons"]["A1_vs_A0"]["wins"], 1)
        self.assertEqual(summary["paired_comparisons"]["A1_vs_A0"]["losses"], 1)
        self.assertEqual(summary["variants"]["A1"]["hit_at_5"], 1.0)
        self.assertEqual(summary["variants"]["A1"]["avg_latency_ms"], 200.0)
        self.assertEqual(summary["variants"]["A1"]["fallback_trigger_rate"], 0.5)
        self.assertEqual(summary["diagnostics"]["A0"]["recall_miss"], 1)

    def test_validate_variant_order_requires_title_index_for_b_group(self):
        with self.assertRaisesRegex(RuntimeError, "expects title_context"):
            validate_variant_order(["B1"], skip_reindex=False)

        validate_variant_order(["B1"], skip_reindex=True)
        validate_variant_order(["A1", "B1", "G1"], skip_reindex=False)


if __name__ == "__main__":
    unittest.main()
