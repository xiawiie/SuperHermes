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
    _stage_metrics,
)
from scripts.rag_qrels import attach_canonical_ids, merge_chunk_qrels


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
        self.assertEqual(VARIANT_CONFIGS["S1_linear"]["env"]["QUERY_PLAN_ENABLED"], "false")
        self.assertEqual(VARIANT_CONFIGS["S2"]["env"]["QUERY_PLAN_ENABLED"], "true")
        self.assertEqual(VARIANT_CONFIGS["S2"]["env"]["DOC_SCOPE_GLOBAL_RESERVE_WEIGHT"], "0.2")
        self.assertEqual(VARIANT_CONFIGS["S2HR"]["env"]["RERANK_PAIR_ENRICHMENT_ENABLED"], "true")
        self.assertEqual(VARIANT_CONFIGS["S3"]["env"]["MILVUS_COLLECTION"], "embeddings_collection_v2")
        self.assertEqual(VARIANT_CONFIGS["S3"]["env"]["EVAL_RETRIEVAL_TEXT_MODE"], "title_context_filename")

    def test_v3_variants_are_isolated_profiles(self):
        self.assertEqual(parse_variants("V3Q,V3F"), ["V3Q", "V3F"])
        self.assertEqual(VARIANT_CONFIGS["V3Q"]["env"]["RAG_INDEX_PROFILE"], "v3_quality")
        self.assertEqual(VARIANT_CONFIGS["V3Q"]["env"]["MILVUS_COLLECTION"], "embeddings_collection_v3_quality")
        self.assertEqual(VARIANT_CONFIGS["V3Q"]["env"]["RERANK_SCORE_FUSION_ENABLED"], "true")
        self.assertEqual(VARIANT_CONFIGS["V3F"]["env"]["RAG_INDEX_PROFILE"], "v3_fast")
        self.assertEqual(VARIANT_CONFIGS["V3F"]["env"]["MILVUS_COLLECTION"], "embeddings_collection_v3_fast")

    def test_gold_variants_are_isolated_from_legacy_and_v3_profiles(self):
        self.assertEqual(parse_variants("GB0,GS1,GS2,GS2H,GS2HR,GS3"), ["GB0", "GS1", "GS2", "GS2H", "GS2HR", "GS3"])
        self.assertEqual(VARIANT_CONFIGS["GB0"]["env"]["RAG_INDEX_PROFILE"], "gold_tc")
        self.assertEqual(VARIANT_CONFIGS["GB0"]["env"]["MILVUS_COLLECTION"], "embeddings_collection_gold_tc")
        self.assertEqual(VARIANT_CONFIGS["GS3"]["env"]["RAG_INDEX_PROFILE"], "gold_tcf")
        self.assertEqual(VARIANT_CONFIGS["GS3"]["env"]["MILVUS_COLLECTION"], "embeddings_collection_gold_tcf")
        self.assertNotEqual(VARIANT_CONFIGS["GS3"]["env"]["MILVUS_COLLECTION"], VARIANT_CONFIGS["V3Q"]["env"]["MILVUS_COLLECTION"])

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
                "dataset": PROJECT_ROOT / ".jbeval" / "datasets" / "rag_doc_gold.jsonl",
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

        summary = summarize_results(rows, variants=["B0_legacy", "S1_linear"])
        rendered = render_summary_markdown(summary)

        self.assertIn("| B0_legacy |", rendered)
        self.assertIn("S1_linear_vs_B0_legacy", rendered)
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
            self.assertEqual(set(summary["variants"]), {"GS3", "V3Q"})
            self.assertEqual(summary["variants"]["GS3"]["file_hit_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
