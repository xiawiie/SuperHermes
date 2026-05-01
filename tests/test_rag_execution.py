import unittest

from backend.chat.rag_execution import (
    RagExecutionPolicy,
    RagTurnRequest,
    mark_rag_execution_policy,
    plan_rag_turn,
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


if __name__ == "__main__":
    unittest.main()
