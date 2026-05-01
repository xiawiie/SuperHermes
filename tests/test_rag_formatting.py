import unittest
from unittest.mock import patch

from backend.chat import tools as chat_tools
from backend.chat.agent import _attached_context_payload
from backend.rag.formatting import (
    NO_RELEVANT_DOCUMENTS_MESSAGE,
    format_rag_documents,
    format_rag_tool_response,
)
from backend.rag.pipeline import _format_docs


class RagFormattingTests(unittest.TestCase):
    def test_pipeline_context_uses_shared_chunk_format(self):
        docs = [
            {"filename": "manual.pdf", "page_number": 2, "text": "alpha"},
            {"filename": "guide.pdf", "page_number": 5, "text": "beta"},
        ]
        expected = "[1] manual.pdf (Page 2):\nalpha\n\n---\n\n[2] guide.pdf (Page 5):\nbeta"

        self.assertEqual(format_rag_documents(docs), expected)
        self.assertEqual(_format_docs(docs), expected)

    def test_tool_response_reuses_graph_context_when_present(self):
        docs = [{"filename": "manual.pdf", "page_number": 2, "text": "alpha"}]

        self.assertEqual(
            format_rag_tool_response(docs, context="[1] already formatted"),
            "Retrieved Chunks:\n[1] already formatted",
        )

    def test_tool_response_can_include_compact_retrieval_meta(self):
        docs = [{"filename": "manual.pdf", "page_number": 2, "text": "alpha"}]
        trace = {
            "candidate_strategy_requested": "standard",
            "candidate_strategy_effective": "standard",
            "candidate_strategy_detail": "global_hybrid",
            "rerank_contract_version": "shared-rerank-v2",
            "postprocess_contract_version": "shared-postprocess-v1",
            "rerank_execution_mode": "executed",
        }

        response = format_rag_tool_response(docs, context="[1] already formatted", retrieval_meta=trace)

        self.assertIn("Retrieval Metadata:", response)
        self.assertIn("candidate_strategy_effective=standard", response)
        self.assertIn("rerank_contract_version=shared-rerank-v2", response)
        self.assertIn("Retrieved Chunks:\n[1] already formatted", response)

    def test_tool_response_keeps_empty_result_contract(self):
        self.assertEqual(format_rag_tool_response([]), NO_RELEVANT_DOCUMENTS_MESSAGE)

    def test_search_tool_reuses_graph_context_and_marks_delivery(self):
        chat_tools.get_last_rag_context(clear=True)
        chat_tools.reset_tool_call_guards()
        rag_result = {
            "docs": [{"filename": "manual.pdf", "page_number": 2, "text": "alpha"}],
            "context": "[1] graph formatted context",
            "rag_trace": {
                "retrieval_mode": "hybrid",
                "candidate_strategy_effective": "standard",
                "rerank_contract_version": "shared-rerank-v2",
                "postprocess_contract_version": "shared-postprocess-v1",
                "rerank_execution_mode": "executed",
            },
        }

        with patch("backend.rag.pipeline.run_rag_graph", return_value=rag_result):
            response = chat_tools.search_knowledge_base.invoke({"query": "q"})

        self.assertIn("Retrieved Chunks:\n[1] graph formatted context", response)
        self.assertIn("Retrieval Metadata:", response)
        stored = chat_tools.get_last_rag_context(clear=True)
        self.assertEqual(stored["rag_trace"]["context_delivery_mode"], "tool_response")
        self.assertEqual(stored["rag_trace"]["retrieval_policy"], "optional_tool")
        self.assertEqual(stored["docs"][0]["filename"], "manual.pdf")

    def test_search_tool_call_guard_survives_tool_invoke_context(self):
        chat_tools.get_last_rag_context(clear=True)
        chat_tools.reset_tool_call_guards()
        rag_result = {
            "docs": [{"filename": "manual.pdf", "page_number": 2, "text": "alpha"}],
            "context": "[1] context",
            "rag_trace": {"retrieval_mode": "hybrid"},
        }

        with patch("backend.rag.pipeline.run_rag_graph", return_value=rag_result):
            first = chat_tools.search_knowledge_base.invoke({"query": "q"})
            second = chat_tools.search_knowledge_base.invoke({"query": "q again"})

        self.assertEqual(first, "Retrieved Chunks:\n[1] context")
        self.assertIn("TOOL_CALL_LIMIT_REACHED", second)

    def test_attached_context_payload_marks_direct_delivery(self):
        context, trace = _attached_context_payload(
            {
                "docs": [{"filename": "manual.pdf", "page_number": 2, "text": "alpha"}],
                "context": "[1] direct context",
                "rag_trace": {"retrieval_mode": "hybrid_scoped"},
            }
        )

        self.assertEqual(context, "[1] direct context")
        self.assertEqual(trace["context_delivery_mode"], "system_message")


if __name__ == "__main__":
    unittest.main()
