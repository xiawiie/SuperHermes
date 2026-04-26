import json
import tempfile
import unittest
from pathlib import Path

from scripts.review_rag_qrels import (
    NoopJudge,
    agreement_policy,
    apply_human_decisions,
    apply_llm_policy,
    build_pool_index,
    precheck_row,
    run_llm_review,
)


class ReviewRagQrelsTests(unittest.TestCase):
    def test_precheck_passes_existing_chunk_and_root(self):
        pool = build_pool_index(
            [
                {
                    "chunk_id": "c1",
                    "root_chunk_id": "r1",
                    "filename": "manual.pdf",
                    "page_number": 3,
                    "text": "evidence text",
                }
            ]
        )
        row = {
            "id": "q1",
            "gold_chunk_ids": ["c1"],
            "expected_root_ids": ["r1"],
            "expected_pages": [3],
            "supporting_chunks": [{"anchor_text": "evidence text"}],
        }

        result = precheck_row(row, pool)

        self.assertEqual(result["pre_check_status"], "pre_check_pass")
        self.assertEqual(result["pre_check_reasons"], [])

    def test_precheck_reports_missing_chunk(self):
        pool = build_pool_index([])
        row = {"id": "q1", "gold_chunk_ids": ["missing"], "expected_root_ids": ["r1"]}

        result = precheck_row(row, pool)

        self.assertEqual(result["pre_check_status"], "pre_check_fail")
        self.assertIn("chunk_id_missing", result["pre_check_reasons"])

    def test_llm_policy_requires_full_support_and_coverage(self):
        accepted = {
            "llm_verdict": "accept",
            "llm_confidence": 0.9,
            "support_level": "full_support",
            "claim_coverage": 0.85,
        }
        weak = {**accepted, "claim_coverage": 0.5}

        self.assertEqual(apply_llm_policy(accepted), "llm_approved")
        self.assertEqual(apply_llm_policy(weak), "needs_human_review")
        self.assertEqual(apply_llm_policy({"llm_verdict": "remap"}), "llm_remapped")

    def test_agreement_policy_table(self):
        self.assertEqual(agreement_policy(0.95), "batch_promote")
        self.assertEqual(agreement_policy(0.75), "advisory_only")
        self.assertEqual(agreement_policy(0.60), "human_only")

    def test_apply_human_decisions_writes_v2_1_without_mutating_input(self):
        rows = [{"id": "q1", "gold_chunk_ids": ["old"], "quality": {"review_status": "draft"}}]
        decisions = [
            {
                "qid": "q1",
                "new_chunk_id": "new",
                "review_status": "corrected",
                "review_source": "human",
                "reviewer_notes": "fixed",
            }
        ]

        output = apply_human_decisions(rows, decisions)

        self.assertEqual(rows[0]["gold_chunk_ids"], ["old"])
        self.assertEqual(output[0]["gold_chunk_ids"], ["new"])
        self.assertEqual(output[0]["qrel_version"], "v2.1")
        self.assertEqual(output[0]["quality"]["review_status"], "corrected")

    def test_llm_review_can_run_with_noop_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "qrels.jsonl"
            pool_path = tmp_path / "pool.jsonl"
            review_dir = tmp_path / "reviews"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "q1",
                        "quality": {"alignment_status": "failed"},
                        "gold_chunk_ids": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pool_path.write_text("", encoding="utf-8")

            args = type(
                "Args",
                (),
                {
                    "input": input_path,
                    "chunk_pool": pool_path,
                    "review_dir": review_dir,
                    "scope": "failed",
                    "sample_size": 1,
                    "seed": 1,
                },
            )()

            self.assertEqual(run_llm_review(args, judge=NoopJudge()), 0)
            self.assertTrue((review_dir / "llm_review_suggestions.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
