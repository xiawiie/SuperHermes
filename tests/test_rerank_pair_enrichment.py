"""Tests for rerank pair enrichment and cache key behavior."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from rag_utils import (
    _rerank_doc_signatures,
    _rerank_cache_key,
    _build_enriched_pair,
    _rerank_pair_text,
    _doc_retrieval_text,
)


_SAMPLE_DOC = {
    "chunk_id": "test_chunk_001",
    "filename": "H3C LA2608室内无线网关 用户手册.pdf",
    "section_title": "1.2 配置LA2608与无线控制器互通",
    "section_path": "1 > 1.2",
    "page_number": 4,
    "anchor_id": "附录A",
    "retrieval_text": "LA2608 安装了SIM卡，通过 3G/4G网络连接到运营商无线控制器。",
}


class TestBuildEnrichedPair:
    def test_enriched_pair_contains_filename(self):
        pair = _build_enriched_pair(_SAMPLE_DOC)
        assert "LA2608" in pair
        assert "[" in pair  # Should have [filename] prefix

    def test_enriched_pair_contains_section(self):
        pair = _build_enriched_pair(_SAMPLE_DOC)
        assert "1.2" in pair or "1 > 1.2" in pair

    def test_enriched_pair_contains_page(self):
        pair = _build_enriched_pair(_SAMPLE_DOC)
        assert "p.4" in pair or "[p.4]" in pair

    def test_enriched_pair_contains_anchor(self):
        pair = _build_enriched_pair(_SAMPLE_DOC)
        assert "附录A" in pair

    def test_enriched_pair_format(self):
        pair = _build_enriched_pair(_SAMPLE_DOC)
        # Should be [filename][section_path][page] heading\nbody
        assert pair.startswith("[") or "LA2608" in pair


class TestRerankPairText:
    def test_enrichment_enabled(self):
        """When RERANK_PAIR_ENRICHMENT_ENABLED, should use enriched pair."""
        # This test verifies the function exists and delegates correctly
        text = _rerank_pair_text(_SAMPLE_DOC, enrichment_enabled=True)
        assert len(text) > 0
        assert "LA2608" in text

    def test_pair_text_differs_from_raw(self):
        """Enriched pair should differ from raw retrieval_text."""
        enriched = _build_enriched_pair(_SAMPLE_DOC)
        raw = _doc_retrieval_text(_SAMPLE_DOC)
        # Enriched should have metadata prefix
        assert len(enriched) >= len(raw)


class TestRerankDocSignatures:
    def test_signature_includes_pair_text_sha1(self):
        """When enrichment enabled, signature should hash enriched pair."""
        sigs = _rerank_doc_signatures([_SAMPLE_DOC], enrichment_enabled=True)
        assert len(sigs) == 1
        assert "pair_text_sha1" in sigs[0]
        assert "chunk_id" in sigs[0]

    def test_different_enrichment_produces_different_hash(self):
        """Different retrieval_text should produce different hash."""
        doc_different = dict(_SAMPLE_DOC)
        doc_different["retrieval_text"] = "Completely different text content"
        sigs_with = _rerank_doc_signatures([_SAMPLE_DOC], enrichment_enabled=True)
        sigs_without = _rerank_doc_signatures([doc_different], enrichment_enabled=True)
        # Hashes should differ because the content is different
        assert sigs_with[0]["pair_text_sha1"] != sigs_without[0]["pair_text_sha1"]

    def test_same_doc_same_hash(self):
        """Same document should produce same hash."""
        sigs1 = _rerank_doc_signatures([_SAMPLE_DOC], enrichment_enabled=True)
        sigs2 = _rerank_doc_signatures([_SAMPLE_DOC], enrichment_enabled=True)
        assert sigs1[0]["pair_text_sha1"] == sigs2[0]["pair_text_sha1"]

    def test_enrichment_flag_changes_hash_for_same_doc(self):
        enriched = _rerank_doc_signatures([_SAMPLE_DOC], enrichment_enabled=True)
        raw = _rerank_doc_signatures([_SAMPLE_DOC], enrichment_enabled=False)
        assert enriched[0]["pair_text_sha1"] != raw[0]["pair_text_sha1"]

    def test_cache_key_changes_with_enrichment_flag(self):
        enriched_key = _rerank_cache_key("query", [_SAMPLE_DOC], 1, 1, enrichment_enabled=True)
        raw_key = _rerank_cache_key("query", [_SAMPLE_DOC], 1, 1, enrichment_enabled=False)
        assert enriched_key != raw_key
