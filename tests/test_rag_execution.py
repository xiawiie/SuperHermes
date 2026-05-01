import unittest
from types import SimpleNamespace

from backend.chat.rag_execution import (
    RagExecutionPolicy,
    RagTurnRequest,
    answer_with_rag_context,
    mark_rag_execution_policy,
    plan_rag_turn,
    prepare_rag_answer_messages,
    stream_answer_with_rag_context,
)


class RagExecutionPolicyTests(unittest.TestCase):
    def test_context_files_force_preload_without_unified_flag(self):
        turn = plan_rag_turn(
            RagTurnRequest(user_text="总结附件", context_files=["manual.pdf"], stream=False),
            unified_execution_enabled=False,
        )

        self.assertEqual(turn.policy, RagExecutionPolicy.FORCED_PRELOAD)
        self.assertEqual(turn.delivery_mode, "system_message")
        self.assertEqual(turn.policy_reason, "attached_context_files")

    def test_no_context_defaults_to_optional_tool(self):
        turn = plan_rag_turn(
            RagTurnRequest(user_text="你好", context_files=[], stream=False),
            unified_execution_enabled=False,
        )

        self.assertEqual(turn.policy, RagExecutionPolicy.OPTIONAL_TOOL)
        self.assertEqual(turn.delivery_mode, "tool_response")

    def test_unified_flag_can_preload_obvious_document_question(self):
        turn = plan_rag_turn(
            RagTurnRequest(user_text="根据知识库说明一下配置步骤", context_files=[], stream=False),
            unified_execution_enabled=True,
        )

        self.assertEqual(turn.policy, RagExecutionPolicy.FORCED_PRELOAD)
        self.assertEqual(turn.policy_reason, "document_intent")

    def test_mark_policy_adds_shared_trace_fields(self):
        turn = plan_rag_turn(
            RagTurnRequest(user_text="总结附件", context_files=["manual.pdf"], stream=False),
            unified_execution_enabled=False,
        )

        trace = mark_rag_execution_policy({"retrieval_mode": "hybrid"}, turn)

        self.assertEqual(trace["retrieval_policy"], "forced_preload")
        self.assertEqual(trace["context_delivery_mode"], "system_message")
        self.assertEqual(trace["context_format_version"], "retrieved-chunks-v1")
        self.assertFalse(trace["rag_unified_execution_enabled"])


class FakeModel:
    def __init__(self):
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        return SimpleNamespace(content="model answer")

    async def astream(self, messages):
        self.invoked_with = messages
        yield SimpleNamespace(content="model stream")


class FakeAgent:
    def __init__(self):
        self.invoked_with = None
        self.config = None

    def invoke(self, payload, config=None):
        self.invoked_with = payload
        self.config = config
        return {"messages": [SimpleNamespace(content="agent answer")]}

    async def astream(self, payload, stream_mode=None, config=None):
        self.invoked_with = payload
        self.config = config
        yield SimpleNamespace(content="agent stream"), {"stream_mode": stream_mode}


class RagAnswerExecutionTests(unittest.IsolatedAsyncioTestCase):
    def test_prepare_forced_preload_messages_injects_retrieved_context(self):
        turn = plan_rag_turn(
            RagTurnRequest(user_text="summarize", context_files=["manual.pdf"], stream=False),
            unified_execution_enabled=False,
        )
        messages = [SimpleNamespace(type="human", content="summarize")]

        prepared = prepare_rag_answer_messages(
            messages,
            turn,
            retrieved_context="evidence",
        )

        self.assertEqual(prepared[-1], messages[-1])
        self.assertIn("Retrieved document context", prepared[-2].content)
        self.assertIn("evidence", prepared[-2].content)

    def test_answer_with_rag_context_hides_model_agent_branch_from_callers(self):
        turn = plan_rag_turn(
            RagTurnRequest(user_text="summarize", context_files=["manual.pdf"], stream=False),
            unified_execution_enabled=False,
        )
        model = FakeModel()
        agent = FakeAgent()

        result = answer_with_rag_context(
            messages=[SimpleNamespace(type="human", content="summarize")],
            turn_context=turn,
            retrieved_context="evidence",
            agent_instance=agent,
            model_instance=model,
        )

        self.assertEqual(result.raw_result.content, "model answer")
        self.assertEqual(result.execution_mode, "preloaded_model")
        self.assertIsNotNone(model.invoked_with)
        self.assertIsNone(agent.invoked_with)

    async def test_stream_answer_with_rag_context_uses_same_execution_contract(self):
        turn = plan_rag_turn(
            RagTurnRequest(user_text="hello", context_files=[], stream=True),
            unified_execution_enabled=False,
        )
        model = FakeModel()
        agent = FakeAgent()

        chunks = [
            item
            async for item in stream_answer_with_rag_context(
                messages=[SimpleNamespace(type="human", content="hello")],
                turn_context=turn,
                retrieved_context="",
                agent_instance=agent,
                model_instance=model,
            )
        ]

        self.assertEqual(chunks[0].content, "agent stream")
        self.assertEqual(agent.config, {"recursion_limit": 8})
        self.assertIsNone(model.invoked_with)


if __name__ == "__main__":
    unittest.main()
