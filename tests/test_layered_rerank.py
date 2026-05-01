"""Tests for layered rerank: split_retrieve, L1 scoring, file-aware prefilter."""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Task 1: split_retrieve return shape
# ---------------------------------------------------------------------------


def _make_hit(chunk_id: str, filename: str, distance: float, root: str = "r1", anchor: str = "") -> dict:
    return {
        "chunk_id": chunk_id,
        "filename": filename,
        "text": f"text-{chunk_id}",
        "retrieval_text": f"text-{chunk_id}",
        "file_type": "pdf",
        "page_number": 1,
        "page_start": 1,
        "page_end": 1,
        "parent_chunk_id": root,
        "root_chunk_id": root,
        "chunk_level": 3,
        "chunk_role": "leaf",
        "chunk_idx": 0,
        "section_title": "",
        "section_type": "",
        "section_path": "",
        "anchor_id": anchor,
    }


def test_split_retrieve_returns_dual_scores():
    """split_retrieve merges dense + sparse results with per-path scores."""
    from unittest.mock import MagicMock, patch

    from backend.infra.vector_store.milvus_client import MilvusManager

    # Build mock Milvus search results.
    # pymilvus hits have .get("id"), .get("distance"), .get("entity", {})
    def _mock_hit(base: dict, distance: float, rank_0: int) -> MagicMock:
        entity = {k: v for k, v in base.items()}

        def hit_get(key, default=None):
            if key == "id":
                return rank_0
            if key == "distance":
                return distance
            if key == "entity":
                return entity
            return default

        m = MagicMock()
        m.get = hit_get
        m.entity = entity
        return m

    dense_result = [[
        _mock_hit(_make_hit("f1::p1", "f1.pdf", 0.9), 0.9, 0),
        _mock_hit(_make_hit("f1::p2", "f1.pdf", 0.8), 0.8, 1),
    ]]
    sparse_result = [[
        _mock_hit(_make_hit("f1::p1", "f1.pdf", 0.7), 0.7, 0),
        _mock_hit(_make_hit("f2::p1", "f2.pdf", 0.6), 0.6, 1),
    ]]

    call_count = [0]

    def fake_call_with_reconnect(fn, **kw):
        call_count[0] += 1
        if call_count[0] == 1:
            return dense_result
        return sparse_result

    with patch.object(MilvusManager, "__init__", lambda self: None):
        mgr = MilvusManager()
        mgr.collection_name = "test_col"
        mgr._call_with_reconnect = fake_call_with_reconnect

        results = mgr.split_retrieve(
            dense_embedding=[0.1] * 1024,
            sparse_embedding={"indices": [1, 2], "values": [0.5, 0.3]},
            dense_top_k=80,
            sparse_top_k=80,
        )

    assert len(results) == 3  # f1::p1, f1::p2, f2::p1

    c1 = next(r for r in results if r["chunk_id"] == "f1::p1")
    assert c1["dense_score"] == 0.9
    assert c1["sparse_score"] == 0.7
    assert c1["dense_rank"] == 1
    assert c1["sparse_rank"] == 1
    assert c1["in_dense"] is True
    assert c1["in_sparse"] is True

    c2 = next(r for r in results if r["chunk_id"] == "f1::p2")
    assert c2["dense_score"] == 0.8
    assert c2["sparse_rank"] is None
    assert c2["in_sparse"] is False

    c3 = next(r for r in results if r["chunk_id"] == "f2::p1")
    assert c3["dense_rank"] is None
    assert c3["in_dense"] is False
    assert c3["sparse_score"] == 0.6


# ---------------------------------------------------------------------------
# Task 2: L1 scoring functions
# ---------------------------------------------------------------------------

from backend.rag.layered_rerank import (
    build_l1_candidates,
    file_aggregate_score,
    l1_chunk_score,
    rank_score,
)
from backend.rag.runtime_config import LayeredRerankConfig


class TestRankScore:
    def test_rank_1_returns_1(self):
        assert rank_score(1, 80) == 1.0

    def test_rank_80_returns_0(self):
        assert rank_score(80, 80) == pytest.approx(0.0)

    def test_rank_40_returns_approx_half(self):
        assert rank_score(40, 80) == pytest.approx(1.0 - 39 / 79)

    def test_none_returns_0(self):
        assert rank_score(None, 80) == 0.0


class TestL1ChunkScore:
    def test_both_paths_high_rank(self):
        score = l1_chunk_score(
            dense_rank=1, sparse_rank=1, top_k=80,
            scope_score=0.0, metadata_score=0.0, anchor_score=0.0,
        )
        assert score > 0.6

    def test_scope_boost_adds_015(self):
        score_no_scope = l1_chunk_score(
            dense_rank=40, sparse_rank=None, top_k=80,
            scope_score=0.0, metadata_score=0.0, anchor_score=0.0,
        )
        score_with_scope = l1_chunk_score(
            dense_rank=40, sparse_rank=None, top_k=80,
            scope_score=1.0, metadata_score=0.0, anchor_score=0.0,
        )
        assert score_with_scope - score_no_scope == pytest.approx(0.15)

    def test_all_zero(self):
        score = l1_chunk_score(
            dense_rank=None, sparse_rank=None, top_k=80,
            scope_score=0.0, metadata_score=0.0, anchor_score=0.0,
        )
        assert score == 0.0


class TestFileAggregateScore:
    def test_single_chunk(self):
        # max(0.8) + 0.30 * mean([0.8]) = 0.8 + 0.24 = 1.04
        assert file_aggregate_score([0.8]) == pytest.approx(1.04)

    def test_multiple_chunks(self):
        score = file_aggregate_score([0.9, 0.7, 0.5, 0.3])
        expected = 0.9 + 0.30 * (0.9 + 0.7 + 0.5) / 3
        assert score == pytest.approx(expected)


class TestBuildL1Candidates:
    def _make_candidates(self, n_files=3, chunks_per_file=5):
        candidates = []
        for f in range(n_files):
            for c in range(chunks_per_file):
                candidates.append({
                    "chunk_id": f"file{f}::p{c}",
                    "filename": f"file{f}.pdf",
                    "root_chunk_id": f"file{f}::root",
                    "page_number": c,
                    "anchor_id": "",
                    "section_title": "",
                    "section_path": "",
                    "dense_score": 0.9 - f * 0.1 - c * 0.02,
                    "dense_rank": f * chunks_per_file + c + 1,
                    "sparse_score": 0.8 - f * 0.1 - c * 0.02,
                    "sparse_rank": f * chunks_per_file + c + 1,
                    "in_dense": True,
                    "in_sparse": True,
                    "score": 0.9 - f * 0.1 - c * 0.02,
                })
        return candidates

    def test_output_within_min_max(self):
        candidates = self._make_candidates()
        result = build_l1_candidates(
            candidates,
            scope_matched_files=[],
            anchor_chunk_ids=[],
            min_k=10,
            max_k=15,
        )
        assert 10 <= len(result) <= 15

    def test_scope_files_always_included(self):
        candidates = self._make_candidates()
        result = build_l1_candidates(
            candidates,
            scope_matched_files=["file2.pdf"],
            anchor_chunk_ids=[],
            min_k=5,
            max_k=40,
        )
        filenames = {r["filename"] for r in result}
        assert "file2.pdf" in filenames

    def test_anchor_chunks_always_included(self):
        candidates = self._make_candidates()
        candidates[7]["anchor_id"] = "1.2.3"
        result = build_l1_candidates(
            candidates,
            scope_matched_files=[],
            anchor_chunk_ids=["file1::p2"],
            min_k=5,
            max_k=40,
        )
        chunk_ids = {r["chunk_id"] for r in result}
        assert "file1::p2" in chunk_ids

    def test_explicit_config_controls_scope_cap_without_module_reload(self):
        candidates = self._make_candidates(n_files=2, chunks_per_file=4)
        config = LayeredRerankConfig(
            l1_top_files=0,
            l1_chunks_per_scope_file=1,
            l1_slot_c_max=10,
            l1_min_candidates=1,
            l1_max_candidates=10,
        )

        result = build_l1_candidates(
            candidates,
            scope_matched_files=["file1.pdf"],
            anchor_chunk_ids=[],
            config=config,
        )

        assert [doc["filename"] for doc in result].count("file1.pdf") == 1
