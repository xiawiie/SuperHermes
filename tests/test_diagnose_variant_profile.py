import unittest

from scripts.diagnose_variant_profile import CollectionProfile, build_diagnosis, expected_filenames, variant_env


class DiagnoseVariantProfileTests(unittest.TestCase):
    def test_variant_env_reads_extracted_configs(self):
        env = variant_env("V3F")

        self.assertEqual(env["RAG_INDEX_PROFILE"], "v3_fast")
        self.assertEqual(env["MILVUS_COLLECTION"], "embeddings_collection_v3_fast")

    def test_expected_filenames_deduplicates_normalized_names(self):
        rows = [
            {"expected_files": ["Manual.PDF"]},
            {"gold_files": ["manual.pdf"]},
            {"expected_files": ["Other.pdf"]},
        ]

        self.assertEqual(expected_filenames(rows), ["Manual.PDF", "Other.pdf"])

    def test_clean_profile_passes_core_health_checks(self):
        qrels = [{"expected_files": [f"file-{idx}.pdf"]} for idx in range(60)]
        filenames = {f"file-{idx}.pdf" for idx in range(60)}
        profile = CollectionProfile(True, 60, 100, filenames, {"file-0.pdf": True})
        compare = CollectionProfile(True, 60, 100, filenames, {"file-0.pdf": True})

        result = build_diagnosis(
            variant="V3F",
            compare_to="GS3",
            qrel_records=qrels,
            variant_profile=profile,
            compare_profile=compare,
        )

        self.assertTrue(result["checks"]["collection_exists"])
        self.assertTrue(result["checks"]["document_count_is_60"])
        self.assertTrue(result["checks"]["chunk_count_within_compare_range"])
        self.assertTrue(result["checks"]["qrel_expected_filename_coverage_ge_0_95"])

    def test_missing_collection_recommends_rebuild(self):
        qrels = [{"expected_files": ["manual.pdf"]}]
        missing = CollectionProfile(False, None, None, set(), {})
        compare = CollectionProfile(True, 60, 100, {"manual.pdf"}, {"manual.pdf": True})

        result = build_diagnosis(
            variant="V3F",
            compare_to="GS3",
            qrel_records=qrels,
            variant_profile=missing,
            compare_profile=compare,
        )

        self.assertIn("collection_exists", result["failures"])
        self.assertEqual(result["recommended_action"], "rebuild")


if __name__ == "__main__":
    unittest.main()
