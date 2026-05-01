import argparse
import json
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_rag_matrix import (
    DEFAULT_VARIANTS,
    VARIANT_CONFIGS,
    compare_sample_rank,
    compute_retrieval_metrics,
    evaluate_sample,
    first_relevant_rank,
    parse_variants,
    render_summary_markdown,
    run_saved_results_summary,
    summarize_results,
    validate_eval_dataset_records,
    validate_variant_order,
    _QRELS_NA,
    _build_fingerprint,
    _file_coverage_details,
    _metadata_coverage_report,
    _qrel_coverage_report,
    _summarize_trace,
    _stage_metrics,
)
from scripts.rag_qrels import attach_canonical_ids, merge_chunk_qrels
from scripts.rag_eval.variants import LEGACY_VARIANT_ALIASES, PAIR_DEFINITIONS


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

    def test_id_metrics_deduplicate_normalized_page_qrel_aliases(self):
        docs = [
            {"filename": "Manual.PDF", "page_number": 4, "chunk_id": "c1"},
            {"filename": "manual.pdf", "page_number": 4, "chunk_id": "c2"},
        ]

        metrics = compute_retrieval_metrics(
            docs,
            relevance_judgments=[{"doc_id": "manual.pdf::p4", "relevance": 3}],
            top_k=5,
        )

        self.assertEqual(metrics["id_context_recall_at_5"], 1.0)
        self.assertEqual(metrics["id_context_precision_at_5"], 0.5)
        self.assertEqual(metrics["ndcg_at_5"], 1.0)
        self.assertEqual(metrics["map_at_5"], 1.0)

    def test_positive_context_qrels_do_not_expand_to_page_offset_aliases(self):
        docs = [
            {"filename": "manual.pdf", "page_number": 3, "chunk_id": "p3"},
            {"filename": "manual.pdf", "page_number": 4, "chunk_id": "p4"},
        ]

        metrics = compute_retrieval_metrics(
            docs,
            positive_contexts=[{"file_name": "manual.pdf", "page_number": 4, "relevance": 3}],
            relevance_judgments=[{"doc_id": "manual.pdf::p4", "relevance": 3}],
            top_k=5,
        )

        self.assertEqual(metrics["id_context_recall_at_5"], 1.0)
        self.assertEqual(metrics["id_context_precision_at_5"], 0.5)
        self.assertLessEqual(metrics["map_at_5"], 1.0)

    def test_page_metrics_use_page_start_and_page_end_metadata(self):
        docs = [{"filename": "manual.pdf", "page_start": 4, "page_end": 6, "chunk_id": "c1"}]

        metrics = compute_retrieval_metrics(
            docs,
            expected_files=["manual.pdf"],
            expected_pages=[5],
            top_k=5,
        )

        self.assertTrue(metrics["file_hit_at_5"])
        self.assertTrue(metrics["page_hit_at_5"])
        self.assertTrue(metrics["file_page_hit_at_5"])

    def test_file_page_metric_normalizes_expected_page_refs(self):
        docs = [{"filename": "Manual.PDF", "page_number": 4, "chunk_id": "c1"}]

        metrics = compute_retrieval_metrics(
            docs,
            expected_files=["manual.pdf"],
            expected_page_refs=["manual.pdf::p4"],
            top_k=5,
        )

        self.assertTrue(metrics["file_hit_at_5"])
        self.assertTrue(metrics["page_hit_at_5"])
        self.assertTrue(metrics["file_page_hit_at_5"])
        self.assertEqual(metrics["recall_at_5"], 1.0)

    def test_file_page_metric_is_unscored_without_page_qrel(self):
        docs = [{"filename": "manual.pdf", "page_number": 1, "chunk_id": "c1"}]

        metrics = compute_retrieval_metrics(docs, expected_files=["manual.pdf"], top_k=5)

        self.assertTrue(metrics["file_hit_at_5"])
        self.assertIsNone(metrics["page_hit_at_5"])
        self.assertIsNone(metrics["file_page_hit_at_5"])
        self.assertFalse(metrics["page_qrel_available"])

    def test_filename_coverage_uses_normalized_match_but_preserves_raw_trace(self):
        report = _file_coverage_details(
            {"C:\\docs\\Manual.PDF"},
            {"manual.pdf"},
        )

        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["exact_coverage"], 0.0)
        self.assertEqual(report["file_matches"][0]["raw_expected"], "Manual.PDF")
        self.assertEqual(report["file_matches"][0]["raw_indexed"], "manual.pdf")
        self.assertEqual(report["file_matches"][0]["match_method"], "normalized")

    def test_metadata_coverage_reports_hard_gate_failures(self):
        report = _metadata_coverage_report(
            [
                {"filename": "manual.pdf", "page_number": 1, "retrieval_text": "body"},
                {"filename": "", "page_number": "", "retrieval_text": ""},
            ]
        )

        failed_metrics = {item["metric"] for item in report["hard_gate_failures"]}
        self.assertFalse(report["hard_gate_pass"])
        self.assertIn("filename_non_empty_rate", failed_metrics)
        self.assertIn("page_number_or_page_start_rate", failed_metrics)
        self.assertIn("retrieval_text_non_empty_rate", failed_metrics)

    def test_qrel_coverage_uses_real_gold_fields_without_chunk_root_fabrication(self):
        report = _qrel_coverage_report(
            [
                {
                    "expected_files": ["manual.pdf"],
                    "expected_page_refs": ["manual.pdf::p1"],
                    "positive_contexts": [{"doc_id": "manual.pdf::p1"}],
                    "relevance_judgments": [{"doc_id": "manual.pdf::p1", "relevance": 3}],
                    "gold_chunk_ids": [],
                    "expected_root_ids": [],
                }
            ]
        )

        self.assertEqual(report["file_qrel_coverage"], 1.0)
        self.assertEqual(report["page_qrel_coverage"], 1.0)
        self.assertEqual(report["chunk_qrel_coverage"], 0.0)
        self.assertEqual(report["root_qrel_coverage"], 0.0)

    def test_stage_metrics_report_strict_file_candidate_recall(self):
        meta = {
            "candidates_before_rerank": [
                {"filename": "wrong.pdf", "text": "keyword"},
                {"filename": "manual.pdf", "text": "other"},
            ],
            "candidates_after_rerank": [{"filename": "wrong.pdf", "text": "keyword"}],
            "candidates_after_structure_rerank": [{"filename": "wrong.pdf", "text": "keyword"}],
        }
        expected = {
            "expected_files": ["manual.pdf"],
            "expected_keywords": ["keyword"],
        }

        metrics = _stage_metrics(meta, expected, top_k=5)

        self.assertEqual(metrics["candidate_recall_before_rerank"], 1.0)
        self.assertEqual(metrics["file_candidate_recall_before_rerank"], 1.0)
        self.assertEqual(metrics["file_rank_before_rerank"], 2)
        self.assertEqual(metrics["file_rerank_drop_rate"], 1.0)

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

    def test_canonical_chunk_qrels_match_across_collection_local_ids(self):
        doc = attach_canonical_ids(
            {
                "chunk_id": "collection-local-leaf",
                "root_chunk_id": "collection-local-root",
                "filename": "Manual.PDF",
                "page_number": 4,
                "page_start": 4,
                "page_end": 4,
                "section_path": "设置 / 蓝牙",
                "retrieval_text": "打开蓝牙并保存设置",
            }
        )

        metrics = compute_retrieval_metrics(
            [doc],
            expected_canonical_chunk_ids=[doc["canonical_chunk_id"]],
            expected_canonical_root_ids=[doc["canonical_root_id"]],
            chunk_qrel_match_mode="canonical",
            root_qrel_match_mode="canonical",
            top_k=5,
        )

        self.assertTrue(metrics["chunk_hit_at_5"])
        self.assertTrue(metrics["root_hit_at_5"])
        self.assertEqual(metrics["chunk_mrr"], 1.0)
        self.assertEqual(metrics["root_mrr"], 1.0)
        self.assertTrue(metrics["answer_support_hit_at_5_experimental"])

    def test_external_chunk_qrels_merge_with_conflict_policy(self):
        records = [{"id": "r1", "gold_chunk_ids": ["old"], "expected_root_ids": []}]
        qrels = [
            {
                "id": "r1",
                "gold_chunk_ids": ["new"],
                "expected_root_ids": ["root"],
                "canonical_chunk_ids": ["canon"],
                "quality": {"review_status": "approved", "alignment_status": "aligned"},
            }
        ]

        merged, report = merge_chunk_qrels(
            records,
            qrels,
            conflict_policy="external",
            chunk_match_mode="canonical",
            root_match_mode="strict_id",
            qrel_source="qrels.jsonl",
        )

        self.assertEqual(merged[0]["gold_chunk_ids"], ["new"])
        self.assertEqual(merged[0]["expected_root_ids"], ["root"])
        self.assertEqual(merged[0]["chunk_qrel_match_mode"], "canonical")
        self.assertEqual(report["conflict_count"], 1)
        self.assertEqual(report["review_coverage"], 1.0)

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

    def test_summarize_file_page_uses_same_doc_metric_not_separate_hits(self):
        rows = [
            {
                "sample_id": "s1",
                "variant": "B0_legacy",
                "metrics": {
                    "file_hit_at_5": True,
                    "page_hit_at_5": True,
                    "file_page_hit_at_5": False,
                    "mrr": 0.0,
                },
                "latency_ms": 1.0,
            }
        ]

        summary = summarize_results(rows, variants=["B0_legacy"])

        self.assertEqual(summary["variants"]["B0_legacy"]["file_page_hit_at_5"], 0.0)

    def test_summarize_results_reports_ce_observability(self):
        rows = [
            {
                "sample_id": "s1",
                "variant": "K2",
                "metrics": {
                    "rerank_enabled": True,
                    "rerank_applied": True,
                    "ce_predict_executed": True,
                    "ce_cache_hit": False,
                    "ce_input_count": 30,
                    "ce_latency_ms": 12.5,
                    "mrr": 1.0,
                },
                "trace": {
                    "candidate_strategy_requested": "layered",
                    "candidate_strategy_effective": "layered",
                    "rerank_execution_mode": "executed",
                },
                "latency_ms": 30.0,
            },
            {
                "sample_id": "s2",
                "variant": "K2",
                "metrics": {
                    "rerank_enabled": True,
                    "rerank_applied": False,
                    "ce_predict_executed": False,
                    "ce_cache_hit": True,
                    "ce_input_count": 0,
                    "mrr": 0.0,
                },
                "trace": {
                    "candidate_strategy_requested": "layered",
                    "candidate_strategy_effective": "dense_fallback",
                    "rerank_execution_mode": "skipped_candidate_only",
                },
                "latency_ms": 10.0,
            },
        ]

        summary = summarize_results(rows, variants=["K2"])

        self.assertEqual(summary["variants"]["K2"]["rerank_enabled_rate"], 1.0)
        self.assertEqual(summary["variants"]["K2"]["rerank_applied_rate"], 0.5)
        self.assertEqual(summary["variants"]["K2"]["ce_predict_executed_rate"], 0.5)
        self.assertEqual(summary["variants"]["K2"]["ce_cache_hit_rate"], 0.5)
        self.assertEqual(summary["variants"]["K2"]["avg_ce_input_count"], 15.0)
        self.assertEqual(summary["variants"]["K2"]["p50_ce_latency_ms"], 12.5)
        self.assertEqual(summary["variants"]["K2"]["candidate_strategy_requested_distribution"], {"layered": 2})
        self.assertEqual(summary["variants"]["K2"]["candidate_strategy_effective_distribution"], {"layered": 1, "dense_fallback": 1})
        self.assertEqual(summary["variants"]["K2"]["rerank_execution_mode_distribution"], {"executed": 1, "skipped_candidate_only": 1})

    def test_report_renders_ce_observability_section(self):
        rendered = render_summary_markdown(
            {
                "generated_at": "2026-05-01T00:00:00",
                "sample_rows": 1,
                "variants": {
                    "K2": {
                        "rows": 1,
                        "rerank_enabled_rate": 1.0,
                        "rerank_applied_rate": 1.0,
                        "ce_predict_executed_rate": 1.0,
                        "ce_cache_hit_rate": 0.0,
                        "avg_ce_input_count": 30.0,
                        "p50_ce_latency_ms": 12.5,
                        "p95_ce_latency_ms": 12.5,
                        "candidate_strategy_requested_distribution": {"layered": 1},
                        "candidate_strategy_effective_distribution": {"layered": 1},
                        "rerank_execution_mode_distribution": {"executed": 1},
                    }
                },
                "paired_comparisons": {},
                "diagnostics": {},
            }
        )

        self.assertIn("## Rerank / CE", rendered)
        self.assertIn("CEInputAvg", rendered)
        self.assertIn("| K2 | 1.000 | 1.000 | 1.000 | 0.000 | 30.000 | 12.500 | 12.500 |", rendered)
        self.assertIn("## Candidate Strategy", rendered)
        self.assertIn("| K2 | layered=1 | layered=1 | executed=1 |", rendered)

    def test_summarize_trace_keeps_candidate_only_fallback_fields(self):
        trace = _summarize_trace(
            {
                "fallback_second_pass_mode": "candidate_only",
                "expanded_retrieval_skipped_reason": None,
                "fallback_mode": "candidate_only_merge",
                "initial_candidate_count": 2,
                "expanded_candidate_count": 3,
                "merged_candidate_count": 5,
                "candidate_only_pass_rerank_execution_mode": "skipped_candidate_only",
                "final_rerank_input_count": 5,
                "final_rerank_execution_mode": "executed",
                "fallback_saved_rerank": True,
                "fallback_saved_full_retrievals": 1,
                "candidate_strategy_requested": "layered",
                "candidate_strategy_effective": "layered",
                "rerank_execution_mode": "executed",
            }
        )

        self.assertEqual(trace["fallback_second_pass_mode"], "candidate_only")
        self.assertEqual(trace["fallback_mode"], "candidate_only_merge")
        self.assertEqual(trace["initial_candidate_count"], 2)
        self.assertEqual(trace["expanded_candidate_count"], 3)
        self.assertEqual(trace["merged_candidate_count"], 5)
        self.assertEqual(trace["candidate_only_pass_rerank_execution_mode"], "skipped_candidate_only")
        self.assertEqual(trace["final_rerank_input_count"], 5)
        self.assertEqual(trace["final_rerank_execution_mode"], "executed")
        self.assertTrue(trace["fallback_saved_rerank"])
        self.assertEqual(trace["fallback_saved_full_retrievals"], 1)
        self.assertEqual(trace["candidate_strategy_requested"], "layered")
        self.assertEqual(trace["candidate_strategy_effective"], "layered")
        self.assertEqual(trace["rerank_execution_mode"], "executed")

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
        self.assertEqual(VARIANT_CONFIGS["F1"]["env"]["RAG_FALLBACK_ENABLED"], "true")

    def test_current_quality_profiles_pin_cross_encoder_for_reproducible_eval(self):
        for variant in ("K2", "K3"):
            with self.subTest(variant=variant):
                env = VARIANT_CONFIGS[variant]["env"]
                self.assertEqual(env["RERANK_MODEL"], "BAAI/bge-reranker-v2-m3")
                self.assertEqual(env["RERANK_PROVIDER"], "local")
                self.assertEqual(env["RERANK_TOP_N"], "30")

    def test_layered_eval_variants_use_candidate_strategy_contract(self):
        env = VARIANT_CONFIGS["K2_LAYERED"]["env"]

        self.assertEqual(env["RAG_CANDIDATE_STRATEGY"], "layered")
        self.assertNotIn("LAYERED_" + "RERANK_ENABLED", env)
        self.assertNotIn("L2_" + "CE_DEFAULT_K", env)
        self.assertNotIn("L" + "3_ROOT_WEIGHT", env)

    def test_historical_layered_experiments_collapse_to_single_variant(self):
        historical_aliases = {
            alias: target for alias, target in LEGACY_VARIANT_ALIASES.items() if alias.startswith("EXP_C")
        }
        self.assertEqual(set(historical_aliases.values()), {"K2_LAYERED"})
        self.assertEqual(parse_variants("EXP_C2,EXP_C25,EXP_C3,EXP_C4,EXP_C5,EXP_C7"), ["K2_LAYERED"])

        for alias in historical_aliases:
            with self.subTest(alias=alias):
                self.assertNotIn(alias, VARIANT_CONFIGS)

    def test_profile_fallback_variants_are_fair_and_candidate_only_is_explicit(self):
        self.assertEqual(parse_variants("K2F,K2F_CAND,K3F,K3F_CAND"), ["K2F", "K2F_CAND", "K3F", "K3F_CAND"])

        for base, fallback, candidate_only in (
            ("K2", "K2F", "K2F_CAND"),
            ("K3", "K3F", "K3F_CAND"),
        ):
            with self.subTest(base=base):
                base_env = VARIANT_CONFIGS[base]["env"]
                fallback_env = VARIANT_CONFIGS[fallback]["env"]
                candidate_env = VARIANT_CONFIGS[candidate_only]["env"]

                self.assertEqual(fallback_env["MILVUS_COLLECTION"], base_env["MILVUS_COLLECTION"])
                self.assertEqual(fallback_env["RAG_INDEX_PROFILE"], base_env["RAG_INDEX_PROFILE"])
                self.assertEqual(fallback_env["RERANK_MODEL"], base_env["RERANK_MODEL"])
                self.assertEqual(fallback_env["CONFIDENCE_GATE_ENABLED"], "true")
                self.assertEqual(fallback_env["RAG_FALLBACK_ENABLED"], "true")
                self.assertNotIn("RAG_FALLBACK_CANDIDATE_ONLY", fallback_env)

                self.assertEqual(candidate_env["MILVUS_COLLECTION"], base_env["MILVUS_COLLECTION"])
                self.assertEqual(candidate_env["RAG_FALLBACK_ENABLED"], "true")
                self.assertEqual(candidate_env["RAG_FALLBACK_CANDIDATE_ONLY"], "true")

    def test_v4_final_variant_names_are_available(self):
        self.assertEqual(
            parse_variants("B0_legacy,K1,S2,S2H,S2HR,S3"),
            ["B0_legacy", "K1", "S2", "S2H", "S2HR", "S3"],
        )
        self.assertEqual(parse_variants("S1_linear"), ["K1"])
        self.assertEqual(VARIANT_CONFIGS["B0_legacy"]["env"], {})
        self.assertEqual(VARIANT_CONFIGS["K1"]["env"]["CONFIDENCE_GATE_ENABLED"], "false")
        self.assertEqual(VARIANT_CONFIGS["K1"]["env"]["RAG_FALLBACK_ENABLED"], "false")
        self.assertEqual(VARIANT_CONFIGS["K1"]["env"]["QUERY_PLAN_ENABLED"], "false")
        self.assertEqual(VARIANT_CONFIGS["S2"]["env"]["QUERY_PLAN_ENABLED"], "true")
        self.assertEqual(VARIANT_CONFIGS["S2"]["env"]["DOC_SCOPE_GLOBAL_RESERVE_WEIGHT"], "0.2")
        self.assertEqual(VARIANT_CONFIGS["S2HR"]["env"]["RERANK_PAIR_ENRICHMENT_ENABLED"], "true")
        self.assertEqual(VARIANT_CONFIGS["S3"]["env"]["MILVUS_COLLECTION"], "embeddings_collection_v2")
        self.assertEqual(VARIANT_CONFIGS["S3"]["env"]["EVAL_RETRIEVAL_TEXT_MODE"], "title_context_filename")

    def test_v3_variants_are_isolated_profiles(self):
        self.assertEqual(parse_variants("K2,V3F"), ["K2", "V3F"])
        self.assertEqual(parse_variants("V3Q,V3Q_OPT"), ["K2", "K3"])
        self.assertEqual(parse_variants("K2,V3Q,K3,V3Q_OPT"), ["K2", "K3"])
        self.assertEqual(VARIANT_CONFIGS["K2"]["env"]["RAG_INDEX_PROFILE"], "v3_quality")
        self.assertEqual(VARIANT_CONFIGS["K2"]["env"]["MILVUS_COLLECTION"], "embeddings_collection_v3_quality")
        self.assertEqual(VARIANT_CONFIGS["K2"]["env"]["RERANK_SCORE_FUSION_ENABLED"], "true")
        self.assertEqual(VARIANT_CONFIGS["V3F"]["env"]["RAG_INDEX_PROFILE"], "v3_fast")
        self.assertEqual(VARIANT_CONFIGS["V3F"]["env"]["MILVUS_COLLECTION"], "embeddings_collection_v3_fast")

    def test_short_profile_variants_are_public_configs_and_legacy_names_are_aliases(self):
        self.assertEqual(parse_variants("K1,K2,K3"), ["K1", "K2", "K3"])
        self.assertEqual(LEGACY_VARIANT_ALIASES["S1_linear"], "K1")
        self.assertEqual(LEGACY_VARIANT_ALIASES["V3Q"], "K2")
        self.assertEqual(LEGACY_VARIANT_ALIASES["V3Q_OPT"], "K3")
        self.assertEqual(parse_variants("V3Q_LAYERED,EXP_C5"), ["K2_LAYERED"])
        self.assertNotIn("S1_linear", VARIANT_CONFIGS)
        self.assertNotIn("V3Q", VARIANT_CONFIGS)
        self.assertNotIn("V3Q_OPT", VARIANT_CONFIGS)
        self.assertNotIn("EXP_C5", VARIANT_CONFIGS)
        self.assertIn("K2_LAYERED", VARIANT_CONFIGS)
        self.assertEqual(VARIANT_CONFIGS["K2"]["legacy_variant"], "V3Q")
        self.assertEqual(VARIANT_CONFIGS["K3"]["legacy_variant"], "V3Q_OPT")
        self.assertEqual(VARIANT_CONFIGS["K2_LAYERED"]["rag_profile"], "K2_LAYERED/I2/M0/A1/fp16")
        self.assertEqual(VARIANT_CONFIGS["K2_LAYERED"]["rag_k"], "K2")

    def test_default_variants_use_short_profile_names(self):
        self.assertEqual(parse_variants(DEFAULT_VARIANTS), ["K2", "K3"])

    def test_pair_definitions_use_canonical_profile_names(self):
        rendered_pairs = "\n".join("|".join(pair) for pair in PAIR_DEFINITIONS)
        self.assertNotIn("S1_linear", rendered_pairs)
        self.assertNotIn("V3Q", rendered_pairs)
        self.assertNotIn("V3Q_OPT", rendered_pairs)
        self.assertIn(("K2_vs_GS3", "GS3", "K2"), PAIR_DEFINITIONS)
        self.assertIn(("K3_vs_K2", "K2", "K3"), PAIR_DEFINITIONS)

    def test_short_profile_config_hash_matches_historical_alias(self):
        args = type(
            "Args",
            (),
            {
                "dataset": PROJECT_ROOT / "eval" / "datasets" / "rag_doc_gold.jsonl",
                "documents_dir": PROJECT_ROOT.parent / "doc",
                "top_k": 5,
                "mode": "retrieval",
            },
        )()

        variants = parse_variants("K2,V3Q")
        fingerprint = _build_fingerprint(args, variants, reindex_events=[])

        self.assertEqual(variants, ["K2"])
        metadata = fingerprint["variants"]["K2"]
        self.assertEqual(metadata["rag_profile"], "K2/I2/M0/A1/fp16")
        self.assertEqual(metadata["rag_k"], "K2")
        self.assertEqual(metadata["rag_i"], "I2")
        self.assertEqual(metadata["rag_m"], "M0")
        self.assertEqual(metadata["rag_a"], "A1")
        self.assertEqual(metadata["rag_dtype"], "fp16")
        self.assertEqual(metadata["legacy_variant"], "V3Q")
        self.assertEqual(metadata["collection"], "embeddings_collection_v3_quality")
        self.assertTrue(str(metadata["bm25_state_path"]).endswith("bm25_state_v3_quality.json"))
        self.assertEqual(metadata["retrieval_text_mode"], "title_context_filename")
        self.assertEqual(metadata["rerank_torch_dtype"], "float16")

    def test_profile_fingerprint_includes_reranker_identity(self):
        args = type(
            "Args",
            (),
            {
                "dataset": PROJECT_ROOT / "eval" / "datasets" / "rag_doc_gold.jsonl",
                "documents_dir": PROJECT_ROOT.parent / "doc",
                "top_k": 5,
                "mode": "retrieval",
            },
        )()

        fingerprint = _build_fingerprint(args, ["K2"], reindex_events=[])

        resolved = fingerprint["variants"]["K2"]["resolved_config"]
        self.assertEqual(resolved["rerank_model"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(resolved["rerank_provider"], "local")

    def test_gold_variants_are_isolated_from_legacy_and_v3_profiles(self):
        self.assertEqual(parse_variants("GB0,GS1,GS2,GS2H,GS2HR,GS3"), ["GB0", "GS1", "GS2", "GS2H", "GS2HR", "GS3"])
        self.assertEqual(VARIANT_CONFIGS["GB0"]["env"]["RAG_INDEX_PROFILE"], "gold_tc")
        self.assertEqual(VARIANT_CONFIGS["GB0"]["env"]["MILVUS_COLLECTION"], "embeddings_collection_gold_tc")
        self.assertEqual(VARIANT_CONFIGS["GS3"]["env"]["RAG_INDEX_PROFILE"], "gold_tcf")
        self.assertEqual(VARIANT_CONFIGS["GS3"]["env"]["MILVUS_COLLECTION"], "embeddings_collection_gold_tcf")
        self.assertNotEqual(VARIANT_CONFIGS["GS3"]["env"]["MILVUS_COLLECTION"], VARIANT_CONFIGS["K2"]["env"]["MILVUS_COLLECTION"])

    def test_chunk_root_metrics_render_na_when_qrel_coverage_is_zero(self):
        rows = [
            {
                "sample_id": "s1",
                "variant": "GS3",
                "metrics": {
                    "file_qrel_available": True,
                    "page_qrel_available": True,
                    "chunk_qrel_available": False,
                    "root_qrel_available": False,
                    "file_hit_at_5": True,
                    "file_page_hit_at_5": True,
                    "chunk_hit_at_5": None,
                    "root_hit_at_5": None,
                    "mrr": 1.0,
                },
                "diagnostic_result": {"category": "ok"},
                "latency_ms": 10.0,
            }
        ]

        summary = summarize_results(rows, variants=["GS3"])

        self.assertEqual(summary["variants"]["GS3"]["chunk_qrel_coverage"], 0.0)
        self.assertEqual(summary["variants"]["GS3"]["root_qrel_coverage"], 0.0)
        self.assertEqual(summary["variants"]["GS3"]["chunk_hit_at_5"], _QRELS_NA)
        self.assertEqual(summary["variants"]["GS3"]["root_hit_at_5"], _QRELS_NA)

    def test_report_renders_build_fingerprint_and_coverage(self):
        args = type(
            "Args",
            (),
            {
                "dataset": PROJECT_ROOT / "eval" / "datasets" / "rag_doc_gold.jsonl",
                "documents_dir": PROJECT_ROOT.parent / "doc",
                "top_k": 5,
                "mode": "retrieval",
            },
        )()
        fingerprint = _build_fingerprint(args, ["GS3"], reindex_events=[])
        summary = {
            "generated_at": "2026-04-25T00:00:00",
            "sample_rows": 0,
            "variants": {},
            "paired_comparisons": {},
            "diagnostics": {},
            "build_fingerprint": fingerprint,
            "coverage_preflight": [
                {"type": "qrel_coverage", "file_qrel_coverage": 1.0, "page_qrel_coverage": 1.0, "chunk_qrel_coverage": 0.0, "root_qrel_coverage": 0.0}
            ],
        }

        rendered = render_summary_markdown(summary)

        self.assertIn("## Build Info", rendered)
        self.assertIn("## Coverage Preflight", rendered)
        self.assertEqual(len(fingerprint["variants"]["GS3"]["profile_config_hash"]), 16)

    def test_report_renders_short_profile_name_and_historical_alias(self):
        args = type(
            "Args",
            (),
            {
                "dataset": PROJECT_ROOT / "eval" / "datasets" / "rag_doc_gold.jsonl",
                "documents_dir": PROJECT_ROOT.parent / "doc",
                "top_k": 5,
                "mode": "retrieval",
            },
        )()
        fingerprint = _build_fingerprint(args, ["K2"], reindex_events=[])
        summary = {
            "generated_at": "2026-04-29T00:00:00",
            "sample_rows": 1,
            "variants": {
                "K2": {
                    "rows": 1,
                    "file_hit_at_5": 1.0,
                    "file_page_hit_at_5": 1.0,
                    "mrr": 1.0,
                    "p50_latency_ms": 10.0,
                    "p95_latency_ms": 10.0,
                }
            },
            "paired_comparisons": {},
            "diagnostics": {},
            "build_fingerprint": fingerprint,
        }

        rendered = render_summary_markdown(summary)

        self.assertIn("K2/I2/M0/A1/fp16", rendered)
        self.assertIn("legacy: V3Q", rendered)

    def test_chunk_root_metrics_render_from_qrels(self):
        rows = [
            {
                "sample_id": "s1",
                "variant": "B0_legacy",
                "metrics": {
                    "hit_at_5": False,
                    "mrr": 0.0,
                    "file_hit_at_5": False,
                    "chunk_hit_at_5": False,
                    "root_hit_at_5": True,
                },
                "diagnostic_result": {"category": "file_recall_miss"},
                "latency_ms": 100.0,
            },
            {
                "sample_id": "s1",
                "variant": "S1_linear",
                "metrics": {
                    "hit_at_5": False,
                    "mrr": 0.0,
                    "file_hit_at_5": False,
                    "chunk_hit_at_5": False,
                    "root_hit_at_5": False,
                },
                "diagnostic_result": {"category": "ranking_miss"},
                "latency_ms": 90.0,
            },
        ]

        summary = summarize_results(rows, variants=parse_variants("B0_legacy,S1_linear"))
        rendered = render_summary_markdown(summary)

        self.assertIn("| B0_legacy |", rendered)
        self.assertIn("K1_vs_B0_legacy", rendered)
        self.assertEqual(summary["variants"]["B0_legacy"]["root_hit_at_5"], 1.0)

    def test_answer_eval_metrics_render_when_present(self):
        rows = [
            {
                "sample_id": "s1",
                "variant": "S3",
                "metrics": {
                    "hit_at_5": True,
                    "mrr": 1.0,
                    "file_hit_at_5": True,
                    "faithfulness_score": 0.8,
                    "answer_relevance_score": 0.7,
                    "citation_coverage": 0.5,
                    "answer_eval_error_rate": 0.0,
                },
                "diagnostic_result": {"category": "ok"},
                "latency_ms": 100.0,
            }
        ]

        summary = summarize_results(rows, variants=["S3"])
        rendered = render_summary_markdown(summary)

        self.assertIn("## Answer Evaluation", rendered)
        self.assertEqual(summary["variants"]["S3"]["faithfulness_score"], 0.8)
        self.assertEqual(summary["variants"]["S3"]["answer_relevance_score"], 0.7)
        self.assertEqual(summary["variants"]["S3"]["citation_coverage"], 0.5)

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

    def test_saved_results_summary_derives_variants_and_writes_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows_path = tmp_path / "results.jsonl"
            rows = [
                {
                    "sample_id": "s1",
                    "variant": "GS3",
                    "metrics": {
                        "file_qrel_available": True,
                        "page_qrel_available": True,
                        "file_hit_at_5": True,
                        "file_page_hit_at_5": True,
                        "mrr": 1.0,
                    },
                    "diagnostic_result": {"category": "ok"},
                    "latency_ms": 10.0,
                },
                {
                    "sample_id": "s1",
                    "variant": "V3Q",
                    "metrics": {
                        "file_qrel_available": True,
                        "page_qrel_available": True,
                        "file_hit_at_5": True,
                        "file_page_hit_at_5": False,
                        "mrr": 0.5,
                    },
                    "diagnostic_result": {"category": "page_miss"},
                    "latency_ms": 20.0,
                },
            ]
            rows_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                summarize_results_jsonl=rows_path,
                limit=None,
                variants=DEFAULT_VARIANTS,
                output_root=tmp_path,
                run_id="saved",
                regression_baseline_summary=None,
                regression_fail_on_diff=False,
            )

            run_saved_results_summary(args)

            summary = json.loads((tmp_path / "saved" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(summary["variants"]), {"GS3", "K2"})
            self.assertEqual(summary["variants"]["GS3"]["file_hit_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
