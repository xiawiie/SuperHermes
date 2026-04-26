"""Tests for scoped/global weighted RRF merging in rag_utils._weighted_rrf_merge."""
from __future__ import annotations



from backend.rag.query_plan import QueryPlan
from backend.rag.utils import _apply_filename_boost, _build_filename_filter, _weighted_rrf_merge


def _make_chunk(chunk_id: str, score: float, filename: str = "a.pdf", page: int = 1):
    return {
        "chunk_id": chunk_id,
        "score": score,
        "metadata": {"filename": filename, "page_start": page},
    }


class TestWeightedRRFMerge:
    def test_empty_inputs(self):
        result = _weighted_rrf_merge([], rrf_k=60)
        assert result == []

    def test_scoped_only(self):
        scoped = [_make_chunk("c1", 0.9), _make_chunk("c2", 0.7)]
        result = _weighted_rrf_merge([(scoped, 0.8)], rrf_k=60)
        assert len(result) == 2
        assert result[0]["chunk_id"] == "c1"

    def test_global_only(self):
        global_results = [_make_chunk("c3", 0.8), _make_chunk("c4", 0.6)]
        result = _weighted_rrf_merge([(global_results, 0.2)], rrf_k=60)
        assert len(result) == 2
        assert result[0]["chunk_id"] == "c3"

    def test_merge_deduplicates(self):
        """Same chunk in both scoped and global gets merged score."""
        scoped = [_make_chunk("c1", 0.9)]
        global_results = [_make_chunk("c1", 0.5)]
        result = _weighted_rrf_merge([(scoped, 0.8), (global_results, 0.2)], rrf_k=60)
        ids = [c["chunk_id"] for c in result]
        assert ids.count("c1") == 1

    def test_scoped_weight_advantage(self):
        """Scoped results should generally rank higher with 0.8 weight."""
        scoped = [_make_chunk("c1", 0.6)]
        global_results = [_make_chunk("c2", 0.9)]
        result = _weighted_rrf_merge([(scoped, 0.8), (global_results, 0.2)], rrf_k=60)
        assert result[0]["chunk_id"] == "c1"

    def test_top_k_truncation(self):
        """Result includes all merged docs; caller should slice."""
        scoped = [_make_chunk(f"s{i}", 0.9 - i * 0.01) for i in range(20)]
        global_results = [_make_chunk(f"g{i}", 0.8 - i * 0.01) for i in range(20)]
        result = _weighted_rrf_merge([(scoped, 0.8), (global_results, 0.2)], rrf_k=60)
        assert len(result) == 40  # no dedup, all unique ids
        top5 = result[:5]
        assert len(top5) == 5

    def test_no_overlap_higher_total(self):
        """When a chunk appears in both lists, it should rank higher than in either alone."""
        scoped = [_make_chunk("c1", 0.5)]
        global_results = [_make_chunk("c1", 0.5), _make_chunk("c2", 0.9)]
        result = _weighted_rrf_merge([(scoped, 0.8), (global_results, 0.2)], rrf_k=60)
        assert result[0]["chunk_id"] == "c1"
        assert result[0]["rrf_merged_score"] > 0

    def test_multiple_result_sets(self):
        """Three result sets with different weights."""
        s1 = [_make_chunk("a", 0.9)]
        s2 = [_make_chunk("b", 0.8)]
        s3 = [_make_chunk("a", 0.5)]
        result = _weighted_rrf_merge([(s1, 0.5), (s2, 0.3), (s3, 0.2)], rrf_k=60)
        assert result[0]["chunk_id"] == "a"


class TestFilenameBoost:
    def test_boost_mode_reranks_matched_filename_without_filtering(self):
        plan = QueryPlan(
            raw_query="q",
            semantic_query="q",
            clean_query="q",
            matched_files=[("target.pdf", 0.70)],
            scope_mode="boost",
            route="scoped_hybrid",
        )
        candidates = [
            _make_chunk("other", 0.95, filename="other.pdf"),
            _make_chunk("target", 0.20, filename="target.pdf"),
        ]

        result = _apply_filename_boost(plan, candidates)

        assert [item["chunk_id"] for item in result] == ["target", "other"]
        assert len(result) == 2
        assert result[0]["filename_boost_applied"] is True
        assert result[0]["filename_boost_score"] > 0

    def test_non_boost_mode_skips_filename_boost(self):
        plan = QueryPlan(
            raw_query="q",
            semantic_query="q",
            clean_query="q",
            matched_files=[("target.pdf", 0.95)],
            scope_mode="filter",
            route="scoped_hybrid",
        )
        candidates = [
            _make_chunk("other", 0.95, filename="other.pdf"),
            _make_chunk("target", 0.20, filename="target.pdf"),
        ]

        result = _apply_filename_boost(plan, candidates)

        assert [item["chunk_id"] for item in result] == ["other", "target"]
        assert "filename_boost_applied" not in result[1]


class TestFilenameFilter:
    def test_filename_filter_escapes_quotes_and_backslashes(self):
        expr = _build_filename_filter(['manual "quoted" \\ path.pdf'])

        assert 'manual \\"quoted\\" \\\\ path.pdf' in expr
        assert expr.startswith("filename in [")
