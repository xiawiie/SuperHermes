# V3Q_LAYERED Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-path CrossEncoder bottleneck with a 4-layer pipeline (L0 dual-path retrieval → L1 file-aware prefilter → L2 adaptive CrossEncoder → L3 weak structure rerank) to reduce P50 latency from ~4.5s to ~1.5s with zero quality regression.

**Architecture:** Add `split_retrieve` to Milvus client for dual dense+sparse+hybrid retrieval. New `layered_rerank.py` orchestrates L1 prefiltering with 3-slot candidate construction. L2 uses existing `rerank_documents` with adaptive K. L3 weakens `apply_structure_rerank` with lower root_weight and CE top-3 protection. Master toggle `LAYERED_RERANK_ENABLED` falls back to current behavior when disabled.

**Tech Stack:** Python, Milvus (pymilvus), sentence_transformers CrossEncoder, existing RAG pipeline.

---

## File Structure

### New files
- `backend/rag/layered_rerank.py` — L0 split retrieval orchestration, L1 prefilter with 3-slot construction, L2 adaptive K selection, L3 CE protection
- `tests/test_layered_rerank.py` — Unit tests for all new logic

### Modified files
- `backend/infra/vector_store/milvus_client.py` — Add `split_retrieve` method
- `backend/rag/utils.py` — Wire `LAYERED_RERANK_ENABLED` toggle, add config env vars, modify `_finish_retrieval_pipeline`
- `backend/rag/context.py` — Modify `apply_structure_rerank` to accept CE scores for top-3 protection
- `scripts/rag_eval/variants.py` — Register `V3Q_LAYERED` variant

---

## Task 1: Add `split_retrieve` to Milvus Client

**Files:**
- Modify: `backend/infra/vector_store/milvus_client.py:344` (after `hybrid_retrieve`)
- Test: `tests/test_layered_rerank.py`

- [ ] **Step 1: Write failing test for `split_retrieve` return shape**

Create `tests/test_layered_rerank.py`:

```python
import pytest


def test_split_retrieve_returns_dual_scores():
    """split_retrieve merges dense + sparse + hybrid results with per-path scores."""
    from unittest.mock import MagicMock, patch

    mock_manager = MagicMock()

    # Mock dense search results
    dense_hits = [
        {"id": "c1", "distance": 0.9, "entity": {
            "chunk_id": "file1::p1", "filename": "f1.pdf",
            "retrieval_text": "text1", "text": "text1",
            "page_number": 1, "page_start": 1, "page_end": 1,
            "parent_chunk_id": "r1", "root_chunk_id": "r1",
            "chunk_level": 3, "chunk_role": "leaf", "chunk_idx": 0,
            "section_title": "", "section_type": "", "section_path": "",
            "anchor_id": "", "file_type": "pdf",
        }},
        {"id": "c2", "distance": 0.8, "entity": {
            "chunk_id": "file1::p2", "filename": "f1.pdf",
            "retrieval_text": "text2", "text": "text2",
            "page_number": 2, "page_start": 2, "page_end": 2,
            "parent_chunk_id": "r1", "root_chunk_id": "r1",
            "chunk_level": 3, "chunk_role": "leaf", "chunk_idx": 1,
            "section_title": "", "section_type": "", "section_path": "",
            "anchor_id": "", "file_type": "pdf",
        }},
    ]

    # Mock sparse search results (overlap on c1, new c3)
    sparse_hits = [
        {"id": "c1", "distance": 0.7, "entity": {
            "chunk_id": "file1::p1", "filename": "f1.pdf",
            "retrieval_text": "text1", "text": "text1",
            "page_number": 1, "page_start": 1, "page_end": 1,
            "parent_chunk_id": "r1", "root_chunk_id": "r1",
            "chunk_level": 3, "chunk_role": "leaf", "chunk_idx": 0,
            "section_title": "", "section_type": "", "section_path": "",
            "anchor_id": "", "file_type": "pdf",
        }},
        {"id": "c3", "distance": 0.6, "entity": {
            "chunk_id": "file2::p1", "filename": "f2.pdf",
            "retrieval_text": "text3", "text": "text3",
            "page_number": 1, "page_start": 1, "page_end": 1,
            "parent_chunk_id": "r2", "root_chunk_id": "r2",
            "chunk_level": 3, "chunk_role": "leaf", "chunk_idx": 0,
            "section_title": "", "section_type": "", "section_path": "",
            "anchor_id": "", "file_type": "pdf",
        }},
    ]

    # Mock hybrid results
    hybrid_hits = [[MagicMock(get=lambda k, d=None: {
        "id": "c1", "distance": 0.85,
        "chunk_id": "file1::p1", "filename": "f1.pdf",
        "retrieval_text": "text1", "text": "text1",
        "page_number": 1, "page_start": 1, "page_end": 1,
        "parent_chunk_id": "r1", "root_chunk_id": "r1",
        "chunk_level": 3, "chunk_role": "leaf", "chunk_idx": 0,
        "section_title": "", "section_type": "", "section_path": "",
        "anchor_id": "", "file_type": "pdf",
    }.get(k, d))]]

    mock_manager._call_with_reconnect = MagicMock(side_effect=lambda fn, **kw: fn(MagicMock(
        search=MagicMock(return_value=[[MagicMock(get=lambda k, d=None: h.get(k, d), entity=h.get("entity", {}), distance=h.get("distance", 0)) for h in dense_hits]] if "dense" in str(kw) else [[MagicMock(get=lambda k, d=None: h.get(k, d), entity=h.get("entity", {}), distance=h.get("distance", 0)) for h in sparse_hits]]),
        hybrid_search=MagicMock(return_value=hybrid_hits),
    )))

    from backend.infra.vector_store.milvus_client import MilvusManager
    with patch.object(MilvusManager, '__init__', lambda self: None):
        mgr = MilvusManager()
        mgr.collection_name = "test_col"
        mgr._call_with_reconnect = mock_manager._call_with_reconnect

        results = mgr.split_retrieve(
            dense_embedding=[0.1] * 1024,
            sparse_embedding={"indices": [1, 2], "values": [0.5, 0.3]},
            dense_top_k=80,
            sparse_top_k=80,
        )

    # Should return list of dicts with dual-path scores
    assert len(results) == 3  # c1, c2, c3 deduped

    c1 = next(r for r in results if r["chunk_id"] == "file1::p1")
    assert c1["dense_score"] == 0.9
    assert c1["sparse_score"] == 0.7
    assert c1["dense_rank"] == 1
    assert c1["sparse_rank"] == 1
    assert c1["in_dense"] is True
    assert c1["in_sparse"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/goahe/Desktop/Project/SuperHermes && python -m pytest tests/test_layered_rerank.py::test_split_retrieve_returns_dual_scores -v`
Expected: FAIL — `MilvusManager` has no `split_retrieve`

- [ ] **Step 3: Implement `split_retrieve`**

Add to `backend/infra/vector_store/milvus_client.py` after line 344 (after `hybrid_retrieve`):

```python
def split_retrieve(
    self,
    dense_embedding: list[float],
    sparse_embedding: dict,
    dense_top_k: int = 80,
    sparse_top_k: int = 80,
    search_ef: int = 64,
    sparse_drop_ratio: float = 0.2,
    filter_expr: str = "",
) -> list[dict]:
    """Retrieve via separate dense and sparse searches, returning per-path scores."""
    _ensure_pymilvus()
    output_fields = _RETRIEVAL_OUTPUT_FIELDS
    dense_ef = _effective_hnsw_ef(search_ef, dense_top_k)

    # Dense search
    dense_results = self._call_with_reconnect(
        lambda client: client.search(
            collection_name=self.collection_name,
            data=[dense_embedding],
            anns_field="dense_embedding",
            search_params={"metric_type": "IP", "params": {"ef": dense_ef}},
            limit=dense_top_k,
            output_fields=output_fields,
            filter=filter_expr,
        ),
        operation_name="dense_split_search",
    )

    # Sparse search
    sparse_results = self._call_with_reconnect(
        lambda client: client.search(
            collection_name=self.collection_name,
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            search_params={"metric_type": "IP", "params": {"drop_ratio_search": sparse_drop_ratio}},
            limit=sparse_top_k,
            output_fields=output_fields,
            filter=filter_expr,
        ),
        operation_name="sparse_split_search",
    )

    # Format and index by chunk_id
    pool: dict[str, dict] = {}

    for hits in dense_results:
        for rank_0, hit in enumerate(hits):
            entity = hit.get("entity", {})
            cid = entity.get("chunk_id", "")
            if not cid:
                continue
            pool[cid] = {
                "id": hit.get("id"),
                "chunk_id": cid,
                "text": entity.get("text", ""),
                "retrieval_text": entity.get("retrieval_text", ""),
                "filename": entity.get("filename", ""),
                "file_type": entity.get("file_type", ""),
                "page_number": entity.get("page_number", 0),
                "page_start": entity.get("page_start", entity.get("page_number", 0)),
                "page_end": entity.get("page_end", entity.get("page_number", 0)),
                "parent_chunk_id": entity.get("parent_chunk_id", ""),
                "root_chunk_id": entity.get("root_chunk_id", ""),
                "chunk_level": entity.get("chunk_level", 0),
                "chunk_role": entity.get("chunk_role", ""),
                "chunk_idx": entity.get("chunk_idx", 0),
                "section_title": entity.get("section_title", ""),
                "section_type": entity.get("section_type", ""),
                "section_path": entity.get("section_path", ""),
                "anchor_id": entity.get("anchor_id", ""),
                "dense_score": hit.get("distance", 0.0),
                "dense_rank": rank_0 + 1,
                "in_dense": True,
                "sparse_score": 0.0,
                "sparse_rank": None,
                "in_sparse": False,
                "score": hit.get("distance", 0.0),
            }

    for hits in sparse_results:
        for rank_0, hit in enumerate(hits):
            entity = hit.get("entity", {})
            cid = entity.get("chunk_id", "")
            if not cid:
                continue
            sparse_rank = rank_0 + 1
            sparse_score = hit.get("distance", 0.0)
            if cid in pool:
                pool[cid]["sparse_score"] = sparse_score
                pool[cid]["sparse_rank"] = sparse_rank
                pool[cid]["in_sparse"] = True
            else:
                pool[cid] = {
                    "id": hit.get("id"),
                    "chunk_id": cid,
                    "text": entity.get("text", ""),
                    "retrieval_text": entity.get("retrieval_text", ""),
                    "filename": entity.get("filename", ""),
                    "file_type": entity.get("file_type", ""),
                    "page_number": entity.get("page_number", 0),
                    "page_start": entity.get("page_start", entity.get("page_number", 0)),
                    "page_end": entity.get("page_end", entity.get("page_number", 0)),
                    "parent_chunk_id": entity.get("parent_chunk_id", ""),
                    "root_chunk_id": entity.get("root_chunk_id", ""),
                    "chunk_level": entity.get("chunk_level", 0),
                    "chunk_role": entity.get("chunk_role", ""),
                    "chunk_idx": entity.get("chunk_idx", 0),
                    "section_title": entity.get("section_title", ""),
                    "section_type": entity.get("section_type", ""),
                    "section_path": entity.get("section_path", ""),
                    "anchor_id": entity.get("anchor_id", ""),
                    "dense_score": 0.0,
                    "dense_rank": None,
                    "in_dense": False,
                    "sparse_score": sparse_score,
                    "sparse_rank": sparse_rank,
                    "in_sparse": True,
                    "score": sparse_score,
                }

    return list(pool.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /c/Users/goahe/Desktop/Project/SuperHermes && python -m pytest tests/test_layered_rerank.py::test_split_retrieve_returns_dual_scores -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /c/Users/goahe/Desktop/Project/SuperHermes
git add backend/infra/vector_store/milvus_client.py tests/test_layered_rerank.py
git commit -m "feat: add split_retrieve to MilvusManager for dual-path retrieval"
```

---

## Task 2: Implement L1 Scoring and Prefilter

**Files:**
- Create: `backend/rag/layered_rerank.py`
- Test: `tests/test_layered_rerank.py`

- [ ] **Step 1: Write failing tests for L1 score normalization and file aggregation**

Append to `tests/test_layered_rerank.py`:

```python
from backend.rag.layered_rerank import (
    rank_score,
    l1_chunk_score,
    file_aggregate_score,
    build_l1_candidates,
)


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

    def test_scope_boost_dominates_when_ranks_low(self):
        score_no_scope = l1_chunk_score(
            dense_rank=40, sparse_rank=None, top_k=80,
            scope_score=0.0, metadata_score=0.0, anchor_score=0.0,
        )
        score_with_scope = l1_chunk_score(
            dense_rank=40, sparse_rank=None, top_k=80,
            scope_score=1.0, metadata_score=0.0, anchor_score=0.0,
        )
        # scope adds 0.15 to the score
        assert score_with_scope - score_no_scope == pytest.approx(0.15)

    def test_all_zero(self):
        score = l1_chunk_score(
            dense_rank=None, sparse_rank=None, top_k=80,
            scope_score=0.0, metadata_score=0.0, anchor_score=0.0,
        )
        assert score == 0.0


class TestFileAggregateScore:
    def test_single_chunk(self):
        assert file_aggregate_score([0.8]) == pytest.approx(0.8)

    def test_multiple_chunks(self):
        score = file_aggregate_score([0.9, 0.7, 0.5, 0.3])
        expected = 0.9 + 0.3 * (0.9 + 0.7 + 0.5) / 3
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/goahe/Desktop/Project/SuperHermes && python -m pytest tests/test_layered_rerank.py::TestRankScore tests/test_layered_rerank.py::TestL1ChunkScore tests/test_layered_rerank.py::TestFileAggregateScore tests/test_layered_rerank.py::TestBuildL1Candidates -v`
Expected: FAIL — module `backend.rag.layered_rerank` does not exist

- [ ] **Step 3: Implement `backend/rag/layered_rerank.py`**

```python
"""Layered rerank: L1 prefilter + L2 adaptive K + L3 weak structure."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Any


# --- Configuration ---
LAYERED_RERANK_ENABLED = os.getenv("LAYERED_RERANK_ENABLED", "false").lower() == "true"

L0_DENSE_TOP_K = int(os.getenv("L0_DENSE_TOP_K", "80"))
L0_SPARSE_TOP_K = int(os.getenv("L0_SPARSE_TOP_K", "80"))
L0_HYBRID_GUARANTEE_K = int(os.getenv("L0_HYBRID_GUARANTEE_K", "20"))
L0_FALLBACK_POOL_MIN = int(os.getenv("L0_FALLBACK_HYBRID_WHEN_POOL_LT", "60"))

L1_TOP_FILES = int(os.getenv("L1_TOP_FILES", "12"))
L1_CHUNKS_PER_FILE_DEFAULT = int(os.getenv("L1_CHUNKS_PER_FILE_DEFAULT", "3"))
L1_CHUNKS_PER_FILE_TOP3 = int(os.getenv("L1_CHUNKS_PER_FILE_TOP3", "4"))
L1_CHUNKS_PER_SCOPE_FILE = int(os.getenv("L1_CHUNKS_PER_SCOPE_FILE", "6"))
L1_CHUNK_MARGIN_THRESHOLD = float(os.getenv("L1_CHUNK_MARGIN_THRESHOLD", "0.05"))
L1_ROUTE_GUARANTEE_K = int(os.getenv("L1_ROUTE_GUARANTEE_K", "5"))
L1_SLOT_C_MAX = int(os.getenv("L1_SLOT_C_MAX", "20"))
L1_SLOT_A_MIN = int(os.getenv("L1_SLOT_A_MIN", "18"))
L1_SLOT_B_MIN = int(os.getenv("L1_SLOT_B_MIN", "6"))
L1_MIN_CANDIDATES = int(os.getenv("L1_MIN_CANDIDATES", "30"))
L1_MAX_CANDIDATES = int(os.getenv("L1_MAX_CANDIDATES", "40"))

L2_CE_HIGH_CONF_K = int(os.getenv("L2_CE_HIGH_CONF_K", "25"))
L2_CE_DEFAULT_K = int(os.getenv("L2_CE_DEFAULT_K", "32"))
L2_CE_LOW_CONF_K = int(os.getenv("L2_CE_LOW_CONF_K", "40"))
L2_CE_TOP_N = int(os.getenv("L2_CE_TOP_N", "15"))
L2_CE_TOP_N_LOW_CONF = int(os.getenv("L2_CE_TOP_N_LOW_CONF", "20"))

L3_ROOT_WEIGHT = float(os.getenv("L3_ROOT_WEIGHT", "0.15"))
L3_SAME_ROOT_CAP_DEFAULT = int(os.getenv("L3_SAME_ROOT_CAP_DEFAULT", "3"))
L3_SAME_ROOT_CAP_SCOPE_QUERY = int(os.getenv("L3_SAME_ROOT_CAP_SCOPE_QUERY", "5"))
L3_SAME_ROOT_CAP_BROAD_QUERY = int(os.getenv("L3_SAME_ROOT_CAP_BROAD_QUERY", "2"))
L3_PROTECT_CE_TOP3 = os.getenv("L3_PROTECT_CE_TOP3", "true").lower() == "true"


# --- L1 Score Functions ---

def rank_score(rank: int | None, top_k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 - (rank - 1) / max(top_k - 1, 1)


def l1_chunk_score(
    dense_rank: int | None,
    sparse_rank: int | None,
    top_k: int,
    scope_score: float = 0.0,
    metadata_score: float = 0.0,
    anchor_score: float = 0.0,
) -> float:
    return (
        0.35 * rank_score(dense_rank, top_k)
        + 0.35 * rank_score(sparse_rank, top_k)
        + 0.15 * scope_score
        + 0.10 * metadata_score
        + 0.05 * anchor_score
    )


def _metadata_score(doc: dict) -> float:
    score = 0.0
    if doc.get("section_title"):
        score += 0.4
    if doc.get("section_path"):
        score += 0.3
    if doc.get("page_number"):
        score += 0.3
    return min(1.0, score)


def _anchor_score(doc: dict, anchor_chunk_ids: set[str]) -> float:
    return 1.0 if doc.get("chunk_id", "") in anchor_chunk_ids else 0.0


def file_aggregate_score(chunk_scores: list[float]) -> float:
    if not chunk_scores:
        return 0.0
    sorted_scores = sorted(chunk_scores, reverse=True)
    top3 = sorted_scores[:3]
    return max(sorted_scores) + 0.30 * (sum(top3) / len(top3))


def _per_file_cap(file_rank: int, is_scope: bool, in_file_margin: float) -> int:
    if is_scope:
        return L1_CHUNKS_PER_SCOPE_FILE
    if file_rank <= 3:
        return L1_CHUNKS_PER_FILE_TOP3
    if in_file_margin < L1_CHUNK_MARGIN_THRESHOLD:
        return L1_CHUNKS_PER_FILE_TOP3
    return L1_CHUNKS_PER_FILE_DEFAULT


def build_l1_candidates(
    candidates: list[dict],
    scope_matched_files: list[str],
    anchor_chunk_ids: list[str],
    min_k: int | None = None,
    max_k: int | None = None,
) -> list[dict]:
    """Build L1 candidate set using 3-slot architecture."""
    min_k = min_k or L1_MIN_CANDIDATES
    max_k = max_k or L1_MAX_CANDIDATES
    anchor_set = set(anchor_chunk_ids)
    scope_set = set(scope_matched_files)
    top_k = L0_DENSE_TOP_K

    # Compute L1 scores for all candidates
    scored = []
    for doc in candidates:
        s = l1_chunk_score(
            dense_rank=doc.get("dense_rank"),
            sparse_rank=doc.get("sparse_rank"),
            top_k=top_k,
            scope_score=doc.get("doc_scope_match_score", 0.0),
            metadata_score=_metadata_score(doc),
            anchor_score=_anchor_score(doc, anchor_set),
        )
        doc = dict(doc)
        doc["l1_score"] = s
        scored.append(doc)

    # --- Slot C: Guarantee (anchor + scope) ---
    slot_c: list[dict] = []
    seen_c: set[str] = set()

    # Anchor matches
    for doc in scored:
        cid = doc.get("chunk_id", "")
        if cid in anchor_set and cid not in seen_c:
            slot_c.append(doc)
            seen_c.add(cid)

    # Scope file top chunks
    scope_by_file: dict[str, list[dict]] = defaultdict(list)
    for doc in scored:
        fn = doc.get("filename", "")
        if fn in scope_set:
            scope_by_file[fn].append(doc)

    for fn, docs in scope_by_file.items():
        docs.sort(key=lambda d: d["l1_score"], reverse=True)
        for doc in docs[:L1_CHUNKS_PER_SCOPE_FILE]:
            cid = doc.get("chunk_id", "")
            if cid not in seen_c:
                slot_c.append(doc)
                seen_c.add(cid)

    slot_c = slot_c[:L1_SLOT_C_MAX]

    # --- Slot A: File-aware ---
    by_file: dict[str, list[dict]] = defaultdict(list)
    for doc in scored:
        by_file[doc.get("filename", "")].append(doc)

    file_scores: list[tuple[str, float]] = []
    for fn, docs in by_file.items():
        fs = file_aggregate_score([d["l1_score"] for d in docs])
        file_scores.append((fn, fs))
    file_scores.sort(key=lambda x: x[1], reverse=True)

    slot_a: list[dict] = []
    seen_a: set[str] = set(seen_c)

    for file_rank, (fn, _fs) in enumerate(file_scores[:L1_TOP_FILES], 1):
        docs = by_file[fn]
        docs.sort(key=lambda d: d["l1_score"], reverse=True)
        is_scope = fn in scope_set

        # Compute in-file margin for dynamic cap
        margin = 1.0
        if len(docs) >= 2:
            margin = abs(docs[0]["l1_score"] - docs[min(3, len(docs) - 1)]["l1_score"])

        cap = _per_file_cap(file_rank, is_scope, margin)
        for doc in docs[:cap]:
            cid = doc.get("chunk_id", "")
            if cid not in seen_a:
                slot_a.append(doc)
                seen_a.add(cid)

    # --- Slot B: Route guarantee (dense-only, sparse-only, metadata) ---
    slot_b: list[dict] = []
    seen_b = set(seen_a)

    dense_only = [d for d in scored if d.get("in_dense") and not d.get("in_sparse")]
    sparse_only = [d for d in scored if d.get("in_sparse") and not d.get("in_dense")]
    meta_matched = [d for d in scored if _metadata_score(d) > 0.5]

    for bucket in [dense_only, sparse_only, meta_matched]:
        bucket.sort(key=lambda d: d["l1_score"], reverse=True)
        count = 0
        for doc in bucket:
            cid = doc.get("chunk_id", "")
            if cid not in seen_b and count < L1_ROUTE_GUARANTEE_K:
                slot_b.append(doc)
                seen_b.add(cid)
                count += 1

    # --- Merge and enforce quotas ---
    all_selected = slot_c + slot_a + slot_b

    # Dedupe
    seen_final: set[str] = set()
    deduped: list[dict] = []
    for doc in all_selected:
        cid = doc.get("chunk_id", "")
        if cid not in seen_final:
            deduped.append(doc)
            seen_final.add(cid)

    # Sort by L1 score
    deduped.sort(key=lambda d: d["l1_score"], reverse=True)

    # Enforce min/max
    if len(deduped) > max_k:
        deduped = deduped[:max_k]

    return deduped


# --- L2 Adaptive K ---

def select_ce_k(
    candidates: list[dict],
    scope_mode: str,
    exact_file_match: bool,
    is_ambiguous: bool,
    dense_sparse_disagree: bool,
) -> tuple[int, int]:
    """Return (ce_input_k, ce_top_n)."""
    if is_ambiguous or dense_sparse_disagree:
        return L2_CE_LOW_CONF_K, L2_CE_TOP_N_LOW_CONF

    top_margin = 0.0
    if len(candidates) >= 2:
        top_margin = abs(candidates[0].get("l1_score", 0) - candidates[1].get("l1_score", 0))

    if scope_mode == "filter" and exact_file_match and top_margin > 0.15:
        return L2_CE_HIGH_CONF_K, L2_CE_TOP_N

    # Check dense/sparse agreement
    if len(candidates) >= 2:
        dense_agree = (
            candidates[0].get("dense_rank") is not None
            and candidates[1].get("dense_rank") is not None
        )
        sparse_agree = (
            candidates[0].get("sparse_rank") is not None
            and candidates[1].get("sparse_rank") is not None
        )
        if (dense_agree or sparse_agree) and top_margin > 0.20:
            return L2_CE_DEFAULT_K, L2_CE_TOP_N

    return L2_CE_DEFAULT_K, L2_CE_TOP_N


# --- L3 Helpers ---

def select_root_cap(
    scope_mode: str,
    exact_file_match: bool,
    dominant_root_share: float,
    query_is_broad: bool,
) -> int:
    if scope_mode in ("filter", "boost") and exact_file_match:
        return L3_SAME_ROOT_CAP_SCOPE_QUERY
    if dominant_root_share > 0.8 and query_is_broad:
        return L3_SAME_ROOT_CAP_BROAD_QUERY
    return L3_SAME_ROOT_CAP_DEFAULT
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/goahe/Desktop/Project/SuperHermes && python -m pytest tests/test_layered_rerank.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /c/Users/goahe/Desktop/Project/SuperHermes
git add backend/rag/layered_rerank.py tests/test_layered_rerank.py
git commit -m "feat: add L1 scoring, file-aware prefilter, adaptive K selection"
```

---

## Task 3: Wire L0 + L1 + L2 into Pipeline

**Files:**
- Modify: `backend/rag/utils.py` (lines 55-88 config, lines 166-273 pipeline)
- Test: `tests/test_layered_rerank.py`

- [ ] **Step 1: Add config env vars to `utils.py`**

In `backend/rag/utils.py`, after line 88 (after `RERANK_FUSION_METADATA_WEIGHT`), add:

```python
# Layered rerank configuration
from backend.rag.layered_rerank import LAYERED_RERANK_ENABLED as _LAYERED_RERANK_ENABLED
```

Then in the existing import block at line 39-46, add:

```python
from backend.rag.layered_rerank import (
    build_l1_candidates as _build_l1_candidates,
    select_ce_k as _select_ce_k,
    select_root_cap as _select_root_cap,
    L0_DENSE_TOP_K as _L0_DENSE_TOP_K,
    L0_SPARSE_TOP_K as _L0_SPARSE_TOP_K,
    L0_HYBRID_GUARANTEE_K as _L0_HYBRID_GUARANTEE_K,
    L0_FALLBACK_POOL_MIN as _L0_FALLBACK_POOL_MIN,
    L3_ROOT_WEIGHT as _L3_ROOT_WEIGHT,
)
```

- [ ] **Step 2: Modify `_finish_retrieval_pipeline` to support layered mode**

In `backend/rag/utils.py`, modify the `_finish_retrieval_pipeline` function (starting at line 166). Add the layered branch inside the `try` block, before the existing rerank call at line 186.

The key change: when `LAYERED_RERANK_ENABLED` is true, the function receives pre-split candidates from the caller (already containing `dense_rank`, `sparse_rank`, etc.) and applies L1 prefiltering before reranking.

Replace lines 183-193 with:

```python
    try:
        if _LAYERED_RERANK_ENABLED:
            # --- Layered Rerank Pipeline ---
            stage_start = time.perf_counter()

            # L1: Prefilter
            anchor_chunk_ids = [
                doc.get("chunk_id", "")
                for doc in retrieved
                if doc.get("anchor_id")
            ]
            scope_matched_files = list({
                fn for fn, score in (query_plan.matched_files if query_plan else [])
            }) if query_plan else []

            l1_candidates = _build_l1_candidates(
                retrieved,
                scope_matched_files=scope_matched_files,
                anchor_chunk_ids=anchor_chunk_ids,
            )
            timings["l1_prefilter_ms"] = elapsed_ms(stage_start)

            # L2: Adaptive K
            scope_mode = query_plan.scope_mode if query_plan else "none"
            exact_file_match = len(scope_matched_files) == 1 if scope_mode == "filter" else False
            ce_input_k, ce_top_n = _select_ce_k(
                l1_candidates,
                scope_mode=scope_mode,
                exact_file_match=exact_file_match,
                is_ambiguous=False,
                dense_sparse_disagree=False,
            )
            ce_candidates = l1_candidates[:ce_input_k]

            stage_start = time.perf_counter()
            reranked, rerank_meta = _rerank_documents(query=query, docs=ce_candidates, top_k=ce_top_n)
            timings["rerank_ms"] = elapsed_ms(stage_start)
            rerank_meta["rerank_input_k"] = ce_input_k
            rerank_meta["rerank_top_n"] = ce_top_n
            rerank_meta["l1_candidate_count"] = len(l1_candidates)

            # L3: Weak structure rerank
            stage_start = time.perf_counter()
            root_cap = _select_root_cap(
                scope_mode=scope_mode,
                exact_file_match=exact_file_match,
                dominant_root_share=0.0,
                query_is_broad=False,
            )
            reranked_docs, structure_meta = _apply_structure_rerank(
                docs=reranked, top_k=top_k, root_weight=_L3_ROOT_WEIGHT, same_root_cap=root_cap,
            )
            timings["structure_rerank_ms"] = elapsed_ms(stage_start)
        else:
            # --- Original Pipeline ---
            stage_start = time.perf_counter()
            reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
            timings["rerank_ms"] = elapsed_ms(stage_start)
            if rerank_meta.get("rerank_error"):
                stage_errors.append(_stage_error("rerank", str(rerank_meta.get("rerank_error")), "ranked_candidates"))

            stage_start = time.perf_counter()
            reranked_docs, structure_meta = _apply_structure_rerank(docs=reranked, top_k=top_k)
            timings["structure_rerank_ms"] = elapsed_ms(stage_start)
```

Note: The `_apply_structure_rerank` wrapper (line 516) currently hardcodes `root_weight=STRUCTURE_RERANK_ROOT_WEIGHT` and `same_root_cap=SAME_ROOT_CAP`. We need to modify it to accept overrides. Change the wrapper at line 516:

```python
def _apply_structure_rerank(docs: List[dict], top_k: int, root_weight: float | None = None, same_root_cap: int | None = None) -> Tuple[List[dict], Dict[str, Any]]:
    return _context_apply_structure_rerank(
        docs,
        top_k,
        enabled=STRUCTURE_RERANK_ENABLED,
        root_weight=root_weight if root_weight is not None else STRUCTURE_RERANK_ROOT_WEIGHT,
        same_root_cap=same_root_cap if same_root_cap is not None else SAME_ROOT_CAP,
    )
```

- [ ] **Step 3: Add L0 split retrieval to `retrieve_documents`**

In `retrieve_documents` (line 715), the Milvus calls happen at lines 794/805 (scoped path) and 926 (standard path). We need to add a branch for layered mode that calls `split_retrieve` instead of `hybrid_retrieve`.

After the embedding generation step (around line 760 where `dense_embedding` and `sparse_embedding` are available), add a helper that replaces the Milvus call:

In the scoped path (around line 790), add before `_scoped_retrieve`:

```python
def _layered_scoped_retrieve():
    """L0 split retrieval for layered mode."""
    candidates = _milvus_manager.split_retrieve(
        dense_embedding, sparse_embedding,
        dense_top_k=_L0_DENSE_TOP_K,
        sparse_top_k=_L0_SPARSE_TOP_K,
        search_ef=MILVUS_SEARCH_EF,
        sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
        filter_expr=filename_filter or base_filter or "",
    )
    # Add hybrid guarantee
    if _L0_HYBRID_GUARANTEE_K > 0:
        hybrid = _milvus_manager.hybrid_retrieve(
            dense_embedding, sparse_embedding,
            top_k=_L0_HYBRID_GUARANTEE_K,
            rrf_k=MILVUS_RRF_K,
            search_ef=MILVUS_SEARCH_EF,
            sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
            filter_expr=filename_filter or base_filter or "",
        )
        existing_ids = {c.get("chunk_id") for c in candidates}
        for h in hybrid:
            if h.get("chunk_id") not in existing_ids:
                h["dense_rank"] = None
                h["sparse_rank"] = None
                h["dense_score"] = 0.0
                h["sparse_score"] = 0.0
                h["in_dense"] = False
                h["in_sparse"] = False
                candidates.append(h)
    # Fallback check
    if len(candidates) < _L0_FALLBACK_POOL_MIN:
        extra = _milvus_manager.hybrid_retrieve(
            dense_embedding, sparse_embedding,
            top_k=_L0_DENSE_TOP_K,
            rrf_k=MILVUS_RRF_K,
            search_ef=MILVUS_SEARCH_EF,
            sparse_drop_ratio=MILVUS_SPARSE_DROP_RATIO,
            filter_expr=filename_filter or base_filter or "",
        )
        existing_ids = {c.get("chunk_id") for c in candidates}
        for h in extra:
            if h.get("chunk_id") not in existing_ids:
                h["dense_rank"] = None
                h["sparse_rank"] = None
                h["dense_score"] = 0.0
                h["sparse_score"] = 0.0
                h["in_dense"] = False
                h["in_sparse"] = False
                candidates.append(h)
    return candidates
```

Then in the retrieval dispatch logic, when `_LAYERED_RERANK_ENABLED` is true, use `_layered_scoped_retrieve()` / `_layered_global_retrieve()` instead of the Milvus hybrid calls. The exact integration point depends on whether scoped or standard path is taken.

For the standard path (line 924-940), wrap similarly:

```python
if _LAYERED_RERANK_ENABLED:
    retrieved = _layered_scoped_retrieve()  # reuse same helper with empty filter
else:
    retrieved = _milvus_manager.hybrid_retrieve(...)
```

- [ ] **Step 4: Run existing RAG pipeline tests**

Run: `cd /c/Users/goahe/Desktop/Project/SuperHermes && python -m pytest tests/test_rag_pipeline.py -v --timeout=60`
Expected: All PASS (layered mode is off by default)

- [ ] **Step 5: Commit**

```bash
cd /c/Users/goahe/Desktop/Project/SuperHermes
git add backend/rag/utils.py
git commit -m "feat: wire L0/L1/L2/L3 layered rerank into pipeline"
```

---

## Task 4: Register V3Q_LAYERED Variant

**Files:**
- Modify: `scripts/rag_eval/variants.py:389` (after V3Q_OPT)

- [ ] **Step 1: Add V3Q_LAYERED to VARIANT_CONFIGS**

In `scripts/rag_eval/variants.py`, after the V3Q_OPT entry (ending at line 389), add:

```python
"V3Q_LAYERED": {
    "description": "v3 quality layered: dual-path retrieval + file-aware L1 + adaptive CE K + weak structure",
    "reindex_mode": "title_context_filename",
    "requires_reindex": False,
    "env": {
        "RAG_INDEX_PROFILE": "v3_quality",
        "MILVUS_COLLECTION": V3_QUALITY_COLLECTION,
        "EVAL_RETRIEVAL_TEXT_MODE": "title_context_filename",
        "STRUCTURE_RERANK_ENABLED": "true",
        "RERANK_SCORE_FUSION_ENABLED": "true",
        "RERANK_PAIR_ENRICHMENT_ENABLED": "true",
        "CONFIDENCE_GATE_ENABLED": "false",
        "QUERY_PLAN_ENABLED": "true",
        "HEADING_LEXICAL_ENABLED": "true",
        "HEADING_LEXICAL_WEIGHT": "0.20",
        "MILVUS_RRF_K": "80",
        "MILVUS_SPARSE_DROP_RATIO": "0.1",
        "MILVUS_SEARCH_EF": "160",
        "RAG_CANDIDATE_K": "80",
        "RERANK_TOP_N": "30",
        "LAYERED_RERANK_ENABLED": "true",
        "L0_DENSE_TOP_K": "80",
        "L0_SPARSE_TOP_K": "80",
        "L0_HYBRID_GUARANTEE_K": "20",
        "L0_FALLBACK_HYBRID_WHEN_POOL_LT": "60",
        "L1_TOP_FILES": "12",
        "L1_MAX_CANDIDATES": "40",
        "L1_MIN_CANDIDATES": "30",
        "L2_CE_DEFAULT_K": "32",
        "L2_CE_TOP_N": "15",
        "L3_ROOT_WEIGHT": "0.15",
        "L3_SAME_ROOT_CAP_DEFAULT": "3",
    },
},
```

- [ ] **Step 2: Verify variant loads without errors**

Run: `cd /c/Users/goahe/Desktop/Project/SuperHermes && python -c "from scripts.rag_eval.variants import VARIANT_CONFIGS; print('V3Q_LAYERED' in VARIANT_CONFIGS)"`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
cd /c/Users/goahe/Desktop/Project/SuperHermes
git add scripts/rag_eval/variants.py
git commit -m "feat: register V3Q_LAYERED variant for evaluation"
```

---

## Task 5: Run Baseline + Layered Evaluation

**Files:** No code changes — run eval to validate

- [ ] **Step 1: Run baseline V3Q for comparison**

Run: `cd /c/Users/goahe/Desktop/Project/SuperHermes && python scripts/evaluate_rag_matrix.py --variants V3Q --skip-reindex`
Expected: Completes in ~10min, produces report in `eval/reports/`

- [ ] **Step 2: Run V3Q_LAYERED**

Run: `cd /c/Users/goahe/Desktop/Project/SuperHermes && python scripts/evaluate_rag_matrix.py --variants V3Q_LAYERED`
Expected: Completes with lower latency. Check `summary.md` for:
- File@5 degradation vs V3Q baseline <= 0.3pp
- P50 latency reduction >= 50%

- [ ] **Step 3: Compare results**

Run: `cd /c/Users/goahe/Desktop/Project/SuperHermes && python -c "
import json, glob
reports = sorted(glob.glob('eval/reports/rag-matrix-*/summary.json'))
for p in reports:
    d = json.load(open(p, encoding='utf-8'))
    for v, m in d.get('variants', {}).items():
        print(f\"{v}: File@5={m.get('file_hit_at_5',0):.3f}  FP@5={m.get('file_page_hit_at_5',0):.3f}  P50={m.get('p50_latency_ms',0):.0f}ms\")
"`

Expected output similar to:
```
V3Q: File@5=0.955  FP@5=0.727  P50=4451ms
V3Q_LAYERED: File@5=0.95x  FP@5=0.72x  P50=~1800ms
```

- [ ] **Step 4: Commit results**

```bash
cd /c/Users/goahe/Desktop/Project/SuperHermes
git add eval/reports/
git commit -m "eval: V3Q vs V3Q_LAYERED comparison results"
```

---

## Task 6: Run Experiment Matrix (C2-C7)

**Files:** No code changes — systematic evaluation

- [ ] **Step 1: Create additional experiment variants**

Add experiment variants to `scripts/rag_eval/variants.py` for C2 through C7 (each with progressively different settings as defined in the spec experiment matrix). Each variant reuses `V3Q_LAYERED` base config with specific overrides.

- [ ] **Step 2: Run C2 (split 80/80 + simple fusion)**

Verify L0 split retrieval works independently without file-aware L1.

- [ ] **Step 3: Run C6 (recommended candidate)**

Full pipeline: split + file-aware + adaptive K + weak structure.

- [ ] **Step 4: Compare all results and select production variant**

Tabulate File@5, File+Page@5, P50 latency across all experiments. Select variant meeting quality gates: File@5 degradation <= 0.3pp, P50 reduction >= 50%.

- [ ] **Step 5: Commit final evaluation results**

```bash
cd /c/Users/goahe/Desktop/Project/SuperHermes
git add eval/reports/ scripts/rag_eval/variants.py
git commit -m "eval: complete experiment matrix C2-C7 with results"
```
