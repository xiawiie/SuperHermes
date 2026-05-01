import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import backend.rag.pipeline as rag_pipeline  # noqa: E402
from backend.rag.runtime_config import load_runtime_config  # noqa: E402


class RagPipelineFastPathTests(unittest.TestCase):
    def test_grade_documents_skips_llm_when_retrieval_is_high_confidence(self):
        runtime_config = replace(load_runtime_config({}), fallback_enabled=True)
        state = {
            "question": "设备安装步骤是什么？",
            "context": "安装步骤\n请先连接电源。",
            "rag_trace": {
                "fallback_required": False,
                "confidence_reasons": [],
            },
        }

        with (
            patch("backend.rag.pipeline.load_runtime_config", return_value=runtime_config),
            patch("backend.rag.pipeline._get_grader_model", side_effect=AssertionError("grader should not be called")),
        ):
            result = rag_pipeline.grade_documents_node(state)

        self.assertEqual(result["route"], "generate_answer")
        self.assertEqual(result["rag_trace"]["grade_score"], "skipped_fast_path")

    def test_grade_documents_enters_rewrite_when_retrieval_requires_fallback(self):
        runtime_config = replace(load_runtime_config({}), fallback_enabled=True)
        state = {
            "question": "第三条主要规定了什么？",
            "context": "第一条 为了保护民事主体的合法权益。",
            "rag_trace": {
                "fallback_required": True,
                "confidence_reasons": ["anchor_mismatch"],
            },
        }

        with (
            patch("backend.rag.pipeline.load_runtime_config", return_value=runtime_config),
            patch("backend.rag.pipeline._get_grader_model", side_effect=AssertionError("grader should not be called")),
        ):
            result = rag_pipeline.grade_documents_node(state)

        self.assertEqual(result["route"], "rewrite_question")
        self.assertEqual(result["rag_trace"]["grade_score"], "fallback_triggered")


if __name__ == "__main__":
    unittest.main()
