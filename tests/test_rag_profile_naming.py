import unittest

from backend.rag.profile_naming import (
    canonical_variant_name,
    profile_config_hash,
    resolve_dtype,
    resolve_profile,
    resolve_runtime_dtype,
    resolve_variant_profile,
)


class RagProfileNamingTests(unittest.TestCase):
    def test_k2_resolves_to_v3q_historical_alias(self):
        profile = resolve_profile("K2", rag_i="I2", rag_m="M0", rag_a="A1", dtype="fp16")

        self.assertEqual(profile.profile_key, "K2")
        self.assertEqual(profile.profile_name, "K2/I2/M0/A1/fp16")
        self.assertEqual(profile.historical_alias, "V3Q")
        self.assertEqual(profile.rerank_torch_dtype, "float16")

    def test_k3_resolves_to_v3q_opt_historical_alias(self):
        profile = resolve_profile("K3", rag_i="I2", rag_m="M0", rag_a="A1", dtype="bf16")

        self.assertEqual(profile.profile_key, "K3")
        self.assertEqual(profile.profile_name, "K3/I2/M0/A1/bf16")
        self.assertEqual(profile.historical_alias, "V3Q_OPT")
        self.assertEqual(profile.rerank_torch_dtype, "bfloat16")

    def test_historical_aliases_canonicalize_to_new_names(self):
        self.assertEqual(canonical_variant_name("V3Q"), "K2")
        self.assertEqual(canonical_variant_name("v3q_opt"), "K3")
        self.assertEqual(canonical_variant_name("S1_linear"), "K1")

    def test_variant_profiles_keep_historical_alias_context(self):
        new_profile = resolve_variant_profile("K2")
        old_profile = resolve_variant_profile("V3Q")

        self.assertEqual(new_profile.profile_key, old_profile.profile_key)
        self.assertEqual(new_profile.historical_alias, old_profile.historical_alias)
        self.assertEqual(new_profile.profile_name, old_profile.profile_name)

    def test_profile_metadata_uses_public_report_field_names(self):
        metadata = resolve_profile("K2", rag_i="I2", rag_m="M0", rag_a="A1", dtype="fp16").as_metadata()

        self.assertEqual(metadata["rag_profile"], "K2/I2/M0/A1/fp16")
        self.assertEqual(metadata["rag_k"], "K2")
        self.assertEqual(metadata["rag_i"], "I2")
        self.assertEqual(metadata["rag_m"], "M0")
        self.assertEqual(metadata["rag_a"], "A1")
        self.assertEqual(metadata["rag_dtype"], "fp16")
        self.assertEqual(metadata["legacy_variant"], "V3Q")
        self.assertEqual(metadata["rerank_torch_dtype"], "float16")
        self.assertEqual(metadata["device_request"], "auto")

    def test_dtype_mapping_is_explicit(self):
        self.assertEqual(resolve_dtype("fp16"), "float16")
        self.assertEqual(resolve_dtype("bf16"), "bfloat16")
        self.assertEqual(resolve_dtype("fp32"), "float32")
        self.assertEqual(resolve_dtype("float16"), "float16")

        with self.assertRaisesRegex(ValueError, "unknown RAG dtype"):
            resolve_dtype("int8")

    def test_rag_dtype_takes_priority_over_legacy_dtype(self):
        self.assertEqual(resolve_runtime_dtype("bf16", "float16"), "bfloat16")
        self.assertEqual(resolve_runtime_dtype(None, "float32"), "float32")
        self.assertEqual(resolve_runtime_dtype(None, None), "float16")

    def test_profile_config_hash_is_stable_and_short(self):
        first = profile_config_hash({"b": 2, "a": 1})
        second = profile_config_hash({"a": 1, "b": 2})

        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)


if __name__ == "__main__":
    unittest.main()
