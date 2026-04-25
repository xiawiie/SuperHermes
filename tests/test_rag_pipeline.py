import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import rag_pipeline  # noqa: E402


class FakeStructuredGrader:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(binary_score="yes")


class FakeGrader:
    def __init__(self):
        self.structured = FakeStructuredGrader()

    def with_structured_output(self, schema):
        return self.structured


class RagPipelinePromptTests(unittest.TestCase):
    def test_grade_prompt_keeps_json_example_literal(self):
        grader = FakeGrader()
        state = {
            "question": "What is the conclusion?",
            "context": "The document has a conclusion.",
            "rag_trace": {},
        }

        with (
            patch.object(rag_pipeline, "RAG_FALLBACK_ENABLED", True),
            patch("rag_pipeline._get_grader_model", return_value=grader),
        ):
            result = rag_pipeline.grade_documents_node(state)

        prompt = grader.structured.messages[0]["content"]
        self.assertIn('{"binary_score":"yes"}', prompt)
        self.assertEqual(result["route"], "generate_answer")
        self.assertEqual(result["rag_trace"]["grade_score"], "yes")

    def test_retrieve_expanded_timeout_returns_initial_result(self):
        initial_docs = [{"text": "initial", "filename": "a.pdf", "chunk_id": "c1"}]
        state = {
            "question": "q",
            "docs": initial_docs,
            "context": "initial context",
            "context_files": [],
            "expansion_type": "step_back",
            "expanded_query": "expanded q",
            "rag_trace": {},
            "fallback_deadline": time.perf_counter() + 0.01,
        }

        def slow_retrieve(*args, **kwargs):
            time.sleep(0.05)
            return {"docs": [{"text": "late", "filename": "b.pdf", "chunk_id": "late"}], "meta": {}}

        with patch("rag_pipeline.retrieve_documents", side_effect=slow_retrieve):
            result = rag_pipeline.retrieve_expanded(state)

        self.assertEqual(result["docs"], initial_docs)
        self.assertTrue(result["rag_trace"]["fallback_timed_out"])
        self.assertTrue(result["rag_trace"]["fallback_returned_initial"])
        self.assertEqual(result["rag_trace"]["retrieval_stage"], "initial")

    def test_retrieve_expanded_complex_retrievals_run_in_parallel(self):
        state = {
            "question": "q",
            "docs": [],
            "context": "",
            "context_files": [],
            "expansion_type": "complex",
            "expanded_query": "step query",
            "hypothetical_doc": "hyde query",
            "rag_trace": {},
            "fallback_deadline": time.perf_counter() + 5,
        }

        def slow_retrieve(query, top_k=5, context_files=None):
            time.sleep(0.2)
            return {
                "docs": [{"text": query, "filename": f"{query}.pdf", "chunk_id": query, "score": 0.9}],
                "meta": {
                    "timings": {"total_retrieve_ms": 200.0},
                    "stage_errors": [],
                    "rerank_enabled": False,
                    "rerank_applied": False,
                    "retrieval_mode": "hybrid",
                    "candidate_k": 5,
                },
            }

        started = time.perf_counter()
        with patch("rag_pipeline.retrieve_documents", side_effect=slow_retrieve):
            result = rag_pipeline.retrieve_expanded(state)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.35)
        self.assertEqual(len(result["docs"]), 2)
        self.assertFalse(result["rag_trace"].get("fallback_timed_out", False))
        self.assertEqual(result["rag_trace"]["retrieval_stage"], "expanded")


if __name__ == "__main__":
    unittest.main()
