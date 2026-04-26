import unittest

from scripts.rag_eval.regression import compare_core_summary, extract_core_summary


class RagEvalRegressionTests(unittest.TestCase):
    def test_extract_core_summary_limits_to_stable_metrics(self):
        summary = {
            "variants": {
                "GS3": {
                    "file_hit_at_5": 0.9,
                    "file_page_hit_at_5": 0.7,
                    "generated_noise": "ignored",
                }
            }
        }

        core = extract_core_summary(summary)

        self.assertEqual(core["GS3"]["file_hit_at_5"], 0.9)
        self.assertNotIn("generated_noise", core["GS3"])

    def test_compare_core_summary_reports_metric_drift(self):
        old = {"variants": {"GS3": {"file_hit_at_5": 0.9}}}
        new = {"variants": {"GS3": {"file_hit_at_5": 0.8}}}

        diffs = compare_core_summary(old, new)

        self.assertEqual(diffs[0]["variant"], "GS3")
        self.assertEqual(diffs[0]["field"], "file_hit_at_5")


if __name__ == "__main__":
    unittest.main()
