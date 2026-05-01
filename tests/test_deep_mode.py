import unittest
from dataclasses import replace

from backend.rag.deep_mode import DeepModeRequest, run_deep_mode
from backend.rag.runtime_config import load_runtime_config


def _fake_retrieve(query, context_files=None):
    return {
        "docs": [
            {
                "chunk_id": f"{query}-c1",
                "filename": "manual.pdf",
                "page_number": 2,
                "text": f"evidence for {query}",
            }
        ],
        "context": f"context for {query}",
        "rag_trace": {
            "retrieval_mode": "hybrid",
            "query": query,
            "candidate_strategy_requested": "standard",
            "candidate_strategy_effective": "standard",
            "candidate_strategy_detail": "global_hybrid",
            "rerank_contract_version": "shared-rerank-v2",
            "postprocess_contract_version": "shared-postprocess-v1",
            "rerank_execution_mode": "executed",
        },
    }


class DeepModeTests(unittest.TestCase):
    def test_shadow_mode_collects_evidence_without_answering(self):
        config = replace(load_runtime_config({}), deep_shadow_enabled=True, deep_min_coverage=1.0)

        result = run_deep_mode(
            DeepModeRequest(question="q", subqueries=["q1", "q2"]),
            retrieve=_fake_retrieve,
            config=config,
        )

        self.assertEqual(result.final_answer, "")
        self.assertTrue(result.deep_executed)
        self.assertTrue(result.fallback_to_standard)
        self.assertEqual(result.fallback_reason, "shadow_mode")
        self.assertEqual(result.evidence_coverage, 1.0)
        self.assertEqual(result.rag_trace["subqueries"], ["q1", "q2"])

    def test_shadow_mode_exposes_structured_evidence_and_subquery_traces(self):
        config = replace(load_runtime_config({}), deep_shadow_enabled=True, deep_min_coverage=1.0)

        result = run_deep_mode(
            DeepModeRequest(question="q", subqueries=["q1", "q2"]),
            retrieve=_fake_retrieve,
            config=config,
        )
        payload = result.as_dict()

        self.assertEqual(result.rag_trace["answer_mode"], "shadow")
        self.assertEqual(payload["evidence_by_subquery"]["q1"][0]["filename"], "manual.pdf")
        self.assertEqual(payload["coverage_by_subquery"], {"q1": True, "q2": True})
        self.assertEqual(
            result.rag_trace["retrieval_trace_by_subquery"]["q1"]["rerank_contract_version"],
            "shared-rerank-v2",
        )

    def test_active_mode_requires_citation_verifier(self):
        config = replace(
            load_runtime_config({}),
            deep_active_enabled=True,
            citation_verify_enabled=False,
            deep_min_coverage=1.0,
        )

        result = run_deep_mode(
            DeepModeRequest(question="q", active=True),
            retrieve=_fake_retrieve,
            synthesize=lambda question, evidence, refs: "answer [C1]",
            config=config,
        )

        self.assertTrue(result.fallback_to_standard)
        self.assertEqual(result.fallback_reason, "citation_verifier_disabled")

    def test_active_request_is_still_feature_flag_gated(self):
        config = replace(load_runtime_config({}), deep_active_enabled=False, deep_shadow_enabled=False)

        result = run_deep_mode(
            DeepModeRequest(question="q", active=True),
            retrieve=_fake_retrieve,
            synthesize=lambda question, evidence, refs: "answer [C1]",
            config=config,
        )

        self.assertFalse(result.deep_executed)
        self.assertEqual(result.fallback_reason, "deep_mode_disabled")

    def test_active_mode_returns_answer_only_after_citation_verification(self):
        config = replace(
            load_runtime_config({}),
            deep_active_enabled=True,
            citation_verify_enabled=True,
            deep_min_coverage=1.0,
        )

        result = run_deep_mode(
            DeepModeRequest(question="q", active=True),
            retrieve=_fake_retrieve,
            synthesize=lambda question, evidence, refs: "answer [C1]",
            config=config,
        )

        self.assertEqual(result.final_answer, "answer [C1]")
        self.assertFalse(result.fallback_to_standard)
        self.assertTrue(result.rag_trace["citation_verifier"]["valid"])

    def test_active_mode_downgrades_invalid_citations(self):
        config = replace(
            load_runtime_config({}),
            deep_active_enabled=True,
            citation_verify_enabled=True,
            deep_min_coverage=1.0,
        )

        result = run_deep_mode(
            DeepModeRequest(question="q", active=True),
            retrieve=_fake_retrieve,
            synthesize=lambda question, evidence, refs: "answer [C99]",
            config=config,
        )

        self.assertEqual(result.final_answer, "")
        self.assertTrue(result.fallback_to_standard)
        self.assertEqual(result.fallback_reason, "citation_verification_failed")
        self.assertEqual(result.rag_trace["citation_verifier"]["unknown_refs"], ["C99"])


if __name__ == "__main__":
    unittest.main()
