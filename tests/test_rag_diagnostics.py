import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from rag_diagnostics import classify_failure  # noqa: E402


class RagDiagnosticsTests(unittest.TestCase):
    def test_classifies_final_root_hit_as_ok(self):
        trace = {
            "retrieved_chunks": [
                {"chunk_id": "leaf-1", "root_chunk_id": "root-1", "text": "第二条 民法调整自然人和法人关系。"}
            ],
            "fallback_required": True,
            "confidence_reasons": ["anchor_mismatch"],
        }

        result = classify_failure("第二条是什么？", trace, expected_root_ids=["root-1"])

        self.assertEqual(result["category"], "ok")
        self.assertEqual(result["failed_stage"], "none")

    def test_classifies_low_confidence_without_ground_truth(self):
        trace = {
            "retrieved_chunks": [{"chunk_id": "leaf-1", "root_chunk_id": "root-1"}],
            "fallback_required": True,
            "confidence_reasons": ["weak_margin_and_root"],
        }

        result = classify_failure("问题", trace)

        self.assertEqual(result["category"], "low_confidence")
        self.assertEqual(result["failed_stage"], "confidence_gate")

    def test_requires_candidate_trace_to_split_recall_and_ranking(self):
        trace = {
            "retrieved_chunks": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root"}],
            "fallback_required": False,
            "confidence_reasons": [],
        }

        result = classify_failure("问题", trace, expected_root_ids=["root-1"])

        self.assertEqual(result["category"], "insufficient_trace")

    def test_classifies_recall_miss(self):
        trace = {
            "retrieved_chunks": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root"}],
            "fallback_required": False,
            "confidence_reasons": [],
            "candidates_before_rerank": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root"}],
            "candidates_after_rerank": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root"}],
            "candidates_after_structure_rerank": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root"}],
        }

        result = classify_failure("问题", trace, expected_root_ids=["root-1"])

        self.assertEqual(result["category"], "recall_miss")
        self.assertEqual(result["failed_stage"], "recall")

    def test_classifies_rerank_miss(self):
        trace = {
            "retrieved_chunks": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root"}],
            "fallback_required": False,
            "confidence_reasons": [],
            "candidates_before_rerank": [{"chunk_id": "right", "root_chunk_id": "root-1"}],
            "candidates_after_rerank": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root"}],
            "candidates_after_structure_rerank": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root"}],
        }

        result = classify_failure("问题", trace, expected_root_ids=["root-1"])

        self.assertEqual(result["category"], "ranking_miss")
        self.assertEqual(result["failed_stage"], "rerank")

    def test_uses_text_preview_for_keyword_candidate_matching(self):
        trace = {
            "retrieved_chunks": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root", "text": "其它内容"}],
            "fallback_required": False,
            "confidence_reasons": [],
            "candidates_before_rerank": [
                {"chunk_id": "right", "root_chunk_id": "root-1", "text_preview": "第二条 民法调整自然人和法人关系。"}
            ],
            "candidates_after_rerank": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root"}],
            "candidates_after_structure_rerank": [{"chunk_id": "wrong", "root_chunk_id": "wrong-root"}],
        }

        result = classify_failure("第二条是什么？", trace, expected_keywords=["自然人"])

        self.assertEqual(result["category"], "ranking_miss")
        self.assertEqual(result["failed_stage"], "rerank")

    def test_anchor_text_match_uses_boundary_aware_check(self):
        from rag_diagnostics import _anchor_in_text

        self.assertTrue(_anchor_in_text("第一条", "第一条 为了保护民事主体的合法权益"))
        self.assertTrue(_anchor_in_text("1.2", "1.2 系统配置"))
        self.assertFalse(_anchor_in_text("1.2", "11.2 高级配置"))
        self.assertFalse(_anchor_in_text("一、", "二十一、其他事项"))
        self.assertTrue(_anchor_in_text("一、", "一、总则"))


if __name__ == "__main__":
    unittest.main()
