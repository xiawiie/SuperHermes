import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

from backend.documents.loader import DocumentLoader  # noqa: E402


class DocumentLoaderRetrievalTextTests(unittest.TestCase):
    def setUp(self):
        self.loader = DocumentLoader()

    def test_compose_retrieval_text_prefers_current_heading(self):
        text = self.loader._compose_retrieval_text(
            body="请先连接电源，然后长按开机键三秒。",
            current_title="安装步骤",
            parent_title=None,
        )

        self.assertEqual(
            text,
            "安装步骤\n请先连接电源，然后长按开机键三秒。",
        )

    def test_compose_retrieval_text_uses_parent_for_generic_heading(self):
        text = self.loader._compose_retrieval_text(
            body="请确认通风良好并避免潮湿环境。",
            current_title="注意事项",
            parent_title="安装与部署",
        )

        self.assertEqual(
            text,
            "安装与部署 > 注意事项\n请确认通风良好并避免潮湿环境。",
        )

    def test_compose_retrieval_text_filters_blacklisted_heading(self):
        text = self.loader._compose_retrieval_text(
            body="这里是正文内容。",
            current_title="目录",
            parent_title=None,
        )

        self.assertEqual(text, "这里是正文内容。")

    def test_raw_retrieval_text_mode_keeps_raw_text(self):
        with patch.dict("os.environ", {"EVAL_RETRIEVAL_TEXT_MODE": "raw"}):
            loader = DocumentLoader()

        text = loader._make_retrieval_text(
            raw_text="原始正文",
            body="正文",
            current_title="安装步骤",
            parent_title=None,
        )

        self.assertEqual(text, "原始正文")

    def test_title_context_filename_includes_metadata_prefix(self):
        with patch.dict("os.environ", {"EVAL_RETRIEVAL_TEXT_MODE": "title_context_filename"}):
            loader = DocumentLoader()

        text = loader._make_retrieval_text(
            raw_text="raw",
            body="正文内容",
            current_title="安装步骤",
            parent_title="部署",
            filename="H3C Manual.pdf",
            page_start=7,
            anchor_id="附录A",
        )

        self.assertIn("[文档: H3C Manual]", text)
        self.assertIn("[章节: 安装步骤]", text)
        self.assertIn("[页: 7]", text)
        self.assertIn("[锚点: 附录A]", text)
        self.assertLessEqual(len(text), 4000)

    def test_title_context_filename_truncates_to_milvus_limit(self):
        with patch.dict("os.environ", {"EVAL_RETRIEVAL_TEXT_MODE": "title_context_filename"}):
            loader = DocumentLoader()

        text = loader._make_retrieval_text(
            raw_text="raw",
            body="x" * 5000,
            current_title="安装步骤",
            parent_title=None,
            filename="H3C Manual.pdf",
            page_start=7,
            anchor_id="附录A",
        )

        self.assertLessEqual(len(text), 4000)
        self.assertIn("[truncated]", text)


if __name__ == "__main__":
    unittest.main()
