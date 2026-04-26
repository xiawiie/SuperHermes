import unittest


from backend.rag.trace import TRACE_SIGNATURE_VERSION, golden_trace_signature, trace_text_hash


class RagTraceTests(unittest.TestCase):
    def test_trace_text_hash_normalizes_whitespace(self):
        self.assertEqual(trace_text_hash("a\r\n b"), trace_text_hash("a b"))

    def test_golden_trace_signature_captures_stable_ids(self):
        trace = {
            "retrieval_mode": "hybrid",
            "fallback_required": False,
            "candidate_count_before_rerank": 1,
            "rerank_input_count": 1,
            "candidates_before_rerank": [
                {"chunk_id": "c1", "retrieval_text": "heading\nbody", "filename": "m.pdf", "page_number": 2}
            ],
            "candidates_after_rerank": [{"chunk_id": "c1"}],
            "candidates_after_structure_rerank": [{"chunk_id": "c1", "filename": "m.pdf", "page_number": 2}],
        }

        signature = golden_trace_signature("q1", "GS3", trace)

        self.assertEqual(signature["signature_version"], TRACE_SIGNATURE_VERSION)
        self.assertEqual(signature["candidate_ids_before_rerank"], ["c1"])
        self.assertEqual(signature["final_top5_file_pages"][0]["filename"], "m.pdf")


if __name__ == "__main__":
    unittest.main()
