import unittest

from backend.rag.citations import (
    build_citation_refs,
    format_evidence_with_refs,
    maybe_verify_citation_trace,
    verify_citations,
)


class RagCitationTests(unittest.TestCase):
    def test_build_refs_and_format_evidence_with_stable_ids(self):
        docs = [
            {"chunk_id": "c1", "filename": "manual.pdf", "page_number": 2, "text": "alpha"},
            {"chunk_id": "c2", "filename": "guide.pdf", "page_number": 5, "text": "beta"},
        ]

        refs = build_citation_refs(docs)
        formatted = format_evidence_with_refs(docs)

        self.assertEqual(refs[0].ref_id, "C1")
        self.assertEqual(refs[0].chunk_id, "c1")
        self.assertIn("[C1] manual.pdf (Page 2):\nalpha", formatted)
        self.assertIn("[C2] guide.pdf (Page 5):\nbeta", formatted)

    def test_verify_citations_marks_unknown_ref(self):
        refs = build_citation_refs([{"chunk_id": "c1", "filename": "manual.pdf", "page_number": 2}])

        result = verify_citations("Supported by [C1] but not [C99].", refs)

        self.assertFalse(result.valid)
        self.assertEqual(result.cited_refs, ["C1", "C99"])
        self.assertEqual(result.unknown_refs, ["C99"])

    def test_verify_citations_marks_metadata_mismatch(self):
        refs = build_citation_refs([{"chunk_id": "c1", "filename": "manual.pdf", "page_number": 2}])

        result = verify_citations("See [C1|file=manual.pdf|page=9].", refs)

        self.assertFalse(result.valid)
        self.assertEqual(result.metadata_mismatches[0]["field"], "page_number")
        self.assertEqual(result.metadata_mismatches[0]["expected"], 2)
        self.assertEqual(result.metadata_mismatches[0]["actual"], "9")

    def test_maybe_verify_trace_is_flag_gated_and_trace_only(self):
        trace = {
            "retrieved_chunks": [
                {"chunk_id": "c1", "filename": "manual.pdf", "page_number": 2, "text": "alpha"}
            ]
        }

        disabled = maybe_verify_citation_trace("answer [C99]", trace, enabled=False)
        enabled = maybe_verify_citation_trace("answer [C99]", trace, enabled=True)

        self.assertNotIn("citation_verifier", disabled)
        self.assertEqual(enabled["citation_verifier"]["unknown_refs"], ["C99"])
        self.assertEqual(enabled["citation_verifier"]["refs"][0]["ref_id"], "C1")


if __name__ == "__main__":
    unittest.main()
