import sys
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

        with patch("rag_pipeline._get_grader_model", return_value=grader):
            result = rag_pipeline.grade_documents_node(state)

        prompt = grader.structured.messages[0]["content"]
        self.assertIn('{"binary_score":"yes"}', prompt)
        self.assertEqual(result["route"], "generate_answer")
        self.assertEqual(result["rag_trace"]["grade_score"], "yes")


if __name__ == "__main__":
    unittest.main()
