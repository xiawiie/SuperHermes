import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

import backend.rag.pipeline as rag_pipeline  # noqa: E402
from backend.rag.runtime_config import load_runtime_config  # noqa: E402


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
        runtime_config = replace(load_runtime_config({}), fallback_enabled=True)
        state = {
            "question": "What is the conclusion?",
            "context": "The document has a conclusion.",
            "rag_trace": {},
        }

        with (
            patch("backend.rag.pipeline.load_runtime_config", return_value=runtime_config),
            patch("backend.rag.pipeline._get_grader_model", return_value=grader),
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

        with patch("backend.rag.pipeline.retrieve_documents", side_effect=slow_retrieve):
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
        with patch("backend.rag.pipeline.retrieve_documents", side_effect=slow_retrieve):
            result = rag_pipeline.retrieve_expanded(state)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.35)
        self.assertEqual(len(result["docs"]), 2)
        self.assertFalse(result["rag_trace"].get("fallback_timed_out", False))
        self.assertEqual(result["rag_trace"]["retrieval_stage"], "expanded")

    def test_retrieve_expanded_candidate_only_skips_full_second_retrieval(self):
        runtime_config = replace(
            load_runtime_config({}),
            fallback_candidate_only_enabled=True,
            fallback_expanded_candidate_k=17,
        )
        state = {
            "question": "q",
            "docs": [{"text": "initial", "filename": "a.pdf", "chunk_id": "c1"}],
            "context": "initial context",
            "context_files": [],
            "expansion_type": "step_back",
            "expanded_query": "expanded q",
            "rag_trace": {},
            "fallback_deadline": time.perf_counter() + 5,
        }
        candidate_result = {
            "candidates": [{"text": "expanded", "filename": "b.pdf", "chunk_id": "c2"}],
            "meta": {
                "retrieval_mode": "hybrid",
                "timings": {"total_retrieve_ms": 12.0},
                "stage_errors": [],
                "rerank_execution_mode": "skipped_candidate_only",
            },
        }

        def fake_finish(**kwargs):
            meta = {
                **kwargs["extra_trace"],
                "retrieval_mode": kwargs["retrieval_mode"],
                "rerank_applied": True,
                "rerank_execution_mode": "executed",
                "rerank_input_count": len(kwargs["retrieved"]),
                "timings": {"total_retrieve_ms": 20.0},
                "stage_errors": [],
            }
            return {"docs": kwargs["retrieved"][:1], "meta": meta}

        with (
            patch("backend.rag.pipeline.load_runtime_config", return_value=runtime_config),
            patch("backend.rag.pipeline.retrieve_candidate_pool", return_value=candidate_result) as pool,
            patch("backend.rag.pipeline.retrieve_documents", side_effect=AssertionError("full retrieval should not run")),
            patch("backend.rag.pipeline.finish_retrieval_pipeline", side_effect=fake_finish) as finish,
        ):
            result = rag_pipeline.retrieve_expanded(state)

        pool.assert_called_once()
        self.assertEqual(pool.call_args.kwargs["candidate_k"], 17)
        finish.assert_called_once()
        trace = result["rag_trace"]
        self.assertEqual(trace["fallback_second_pass_mode"], "candidate_only")
        self.assertEqual(trace["fallback_mode"], "candidate_only_merge")
        self.assertEqual(trace["initial_candidate_count"], 1)
        self.assertEqual(trace["expanded_candidate_count"], 1)
        self.assertEqual(trace["merged_candidate_count"], 2)
        self.assertEqual(trace["candidate_only_pass_rerank_execution_mode"], "skipped_candidate_only")
        self.assertEqual(trace["final_rerank_execution_mode"], "executed")
        self.assertEqual(trace["rerank_execution_mode"], "executed")
        self.assertEqual(trace["fallback_saved_full_retrievals"], 1)

    def test_retrieve_expanded_candidate_only_complex_runs_candidate_queries_in_parallel(self):
        runtime_config = replace(load_runtime_config({}), fallback_candidate_only_enabled=True)
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

        def slow_candidate_pool(query, top_k=5, context_files=None, candidate_k=None):
            time.sleep(0.2)
            return {
                "candidates": [{"text": query, "filename": f"{query}.pdf", "chunk_id": query}],
                "meta": {
                    "retrieval_mode": "hybrid",
                    "timings": {"total_retrieve_ms": 200.0},
                    "stage_errors": [],
                    "rerank_execution_mode": "skipped_candidate_only",
                },
            }

        def fake_finish(**kwargs):
            return {
                "docs": kwargs["retrieved"],
                "meta": {
                    **kwargs["extra_trace"],
                    "retrieval_mode": kwargs["retrieval_mode"],
                    "rerank_execution_mode": "executed",
                    "timings": {"total_retrieve_ms": 210.0},
                    "stage_errors": [],
                },
            }

        started = time.perf_counter()
        with (
            patch("backend.rag.pipeline.load_runtime_config", return_value=runtime_config),
            patch("backend.rag.pipeline.retrieve_candidate_pool", side_effect=slow_candidate_pool),
            patch("backend.rag.pipeline.finish_retrieval_pipeline", side_effect=fake_finish),
        ):
            result = rag_pipeline.retrieve_expanded(state)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.35)
        self.assertEqual(len(result["docs"]), 2)
        self.assertEqual(result["rag_trace"]["fallback_second_pass_mode"], "candidate_only")
        self.assertEqual(result["rag_trace"]["candidate_only_pass_rerank_execution_mode"], "skipped_candidate_only")
        self.assertEqual(result["rag_trace"]["final_rerank_execution_mode"], "executed")


if __name__ == "__main__":
    unittest.main()
