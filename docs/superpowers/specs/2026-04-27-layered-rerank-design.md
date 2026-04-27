# V3Q_LAYERED: Dual-Path Retrieval with Layered Rerank

**Date:** 2026-04-27
**Status:** Draft
**Target:** Production API latency reduction with zero quality regression

## Problem

CrossEncoder reranking is the dominant latency bottleneck (84.5% of total, P50=4.3s for 80 candidates). Milvus `hybrid_search` merges dense+sparse internally via RRF, discarding per-path scores and limiting L1 prefilter flexibility.

Current funnel:

```
Milvus hybrid_search → 120-200 candidates (RRF score only)
  → CrossEncoder on 50-80 candidates → 4.3s P50
  → structure_rerank → top5
```

## Solution: Four-Layer Pipeline

```
L0: Dual-path independent retrieval (dense + sparse + hybrid guarantee)
L1: File-aware fusion with high-recall prefilter (3-slot architecture)
L2: CrossEncoder on reduced candidates (adaptive K)
L3: Weak structure rerank with CE protection
```

---

## L0: Dual-Path Independent Retrieval

### What changes

Replace single `hybrid_search` with two independent `search` calls + hybrid safety net. This preserves per-path scores and ranks that Milvus's internal RRF merger discards.

### New method: `split_retrieve`

```
Input:  dense_embedding, sparse_embedding
Calls:  search(dense, top_k=80)  →  each hit gets dense_score, dense_rank
        search(sparse, top_k=80) →  each hit gets sparse_score, sparse_rank
        hybrid_search(top_k=20)  →  safety net (always, not just on fallback)
Merge:  union by chunk_id, each candidate carries:
          dense_score, dense_rank, in_dense
          sparse_score, sparse_rank, in_sparse
          hybrid_score (if present)
Fallback: if dense_count < 60 or sparse_count < 60 or either search fails,
          expand hybrid_search to top_k=80
```

### Candidate pool size

Expected: 100-160 unique candidates (dense and sparse overlap ~40-50%).

### Milvus calls

Two independent `search` calls can be parallelized at the Milvus client level. Expected latency: ~60ms (vs ~40ms for current single hybrid_search). The hybrid guarantee adds minimal overhead since Milvus caches warm results.

### Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `L0_DENSE_TOP_K` | 80 | Covers file_rank_p95 |
| `L0_SPARSE_TOP_K` | 80 | Matches dense for symmetry |
| `L0_ENABLE_PARALLEL_SEARCH` | true | Submit both searches concurrently |
| `L0_HYBRID_GUARANTEE_K` | 20 | Always include hybrid top20 |
| `L0_FALLBACK_HYBRID_WHEN_POOL_LT` | 60 | Expand hybrid if either path is sparse |

---

## L1: File-Aware Fusion Prefilter

### Goal

Reduce L0's 100-160 candidates to 30-40 for CrossEncoder, while maintaining file recall >= 0.99 and chunk recall >= current K=50 baseline.

### Score normalization

All signals normalized to [0, 1] before fusion:

```python
def rank_score(rank: int | None, top_k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 - (rank - 1) / max(top_k - 1, 1)
    # rank=1 → 1.0, rank=40 → ~0.5, rank=80 → 0.0
```

This fixes the critical scale mismatch where 1/(rank+60) ≈ 0.016 coexisted with scope_score ∈ [0, 1].

### Chunk-level L1 score

```python
l1_chunk_score = (
    0.35 * rank_score(dense_rank, L0_DENSE_TOP_K)
    + 0.35 * rank_score(sparse_rank, L0_SPARSE_TOP_K)
    + 0.15 * scope_score          # [0, 1] from QueryPlan file matching
    + 0.10 * metadata_score       # [0, 1] from section_title/anchor_id/path
    + 0.05 * anchor_score         # 0 or 1 from anchor exact match
)
```

### File-level aggregation

```python
file_score = (
    max(chunk_scores)
    + 0.30 * mean(top3_chunk_scores)
    + 0.10 * file_scope_score
)
```

### 3-Slot candidate construction

**Slot C — Guarantee (max 20 candidates):**
- Anchor exact match chunks: all included
- Scope-matched file top chunks: k=6 per scope file
- Priority: anchor > scope

**Slot A — File-aware (min 18 candidates):**
- Rank files by `file_score`, select top 12
- Per-file dynamic chunk cap:
  - Scope-matched file → 6
  - Top-3 ranked file → 4
  - In-file top_chunk_margin < 0.05 → 4 (scores very close)
  - Default → 3
- Select top chunks per file by `l1_chunk_score`

**Slot B — Route guarantee (min 6 candidates):**
- Dense-only high-rank chunks not in A/C: top 5
- Sparse-only high-rank chunks not in A/C: top 5
- Metadata/path/title matched: top 5

**Final selection:**
```python
selected = dedupe(slot_c + slot_a + slot_b)
selected = sort_by_l1_chunk_score(selected)
selected = enforce_min_max(selected, min_k=30, max_k=40)
```

### Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `L1_TOP_FILES` | 12 | Files selected in Slot A |
| `L1_CHUNKS_PER_FILE_DEFAULT` | 3 | Default per-file cap |
| `L1_CHUNKS_PER_FILE_TOP3` | 4 | Top-3 file cap |
| `L1_CHUNKS_PER_SCOPE_FILE` | 6 | Scope-matched file cap |
| `L1_CHUNK_MARGIN_THRESHOLD` | 0.05 | If in-file top chunks are this close, use higher cap |
| `L1_ROUTE_GUARANTEE_K` | 5 | Per sub-slot in Slot B |
| `L1_SLOT_C_MAX` | 20 | Prevent guarantee from dominating |
| `L1_SLOT_A_MIN` | 18 | Ensure file-aware representation |
| `L1_SLOT_B_MIN` | 6 | Ensure route diversity |
| `L1_MIN_CANDIDATES` | 30 | Floor for L2 input |
| `L1_MAX_CANDIDATES` | 40 | Ceiling for L2 input |
| `L1_SCOPE_GUARANTEE` | true | Include scope-matched chunks unconditionally |
| `L1_ANCHOR_GUARANTEE` | true | Include anchor-matched chunks unconditionally |

---

## L2: CrossEncoder with Adaptive K

### Adaptive K selection

Priority order (disagree/ambiguous checked first):

```python
if ambiguous or dense_sparse_disagree:
    ce_k = 40                              # Low confidence: full safety net
elif scope_mode == "filter" and exact_file_match and l1_top_margin > 0.15:
    ce_k = 25                              # High confidence: aggressive compression
elif dense_sparse_agree and l1_top_margin > 0.20:
    ce_k = 32                              # Medium-high confidence
else:
    ce_k = 32                              # Default
```

`dense_sparse_disagree`: top-3 candidates differ between dense and sparse rankings.
`l1_top_margin`: `l1_score[0] - l1_score[1]` after normalization.

### Score fusion (after CE)

All inputs normalized to [0, 1] via per-query min-max before weighting:

```python
final_score = (
    0.75 * ce_score_norm
    + 0.10 * rrf_score_norm
    + 0.10 * scope_score_norm
    + 0.05 * metadata_score_norm
)
```

Output: top 15 (low-confidence queries: top 20).

### Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `L2_CE_HIGH_CONF_K` | 25 | High confidence CE input |
| `L2_CE_DEFAULT_K` | 32 | Default CE input |
| `L2_CE_LOW_CONF_K` | 40 | Low confidence CE input |
| `L2_CE_TOP_N` | 15 | CE output size |
| `L2_CE_TOP_N_LOW_CONF` | 20 | CE output for low confidence |

---

## L3: Weak Structure Rerank with CE Protection

### Changes from current

- `root_weight`: 0.30 → 0.15 (CE score dominates)
- `same_root_cap`: conditional instead of fixed
- CE top-3 protected from demotion

### Conditional root cap

```python
if scope_mode in ("filter", "boost") and exact_file_match:
    same_root_cap = 5          # Scope query: allow multi-chunk from same root
elif dominant_root_share > 0.8 and query_is_broad:
    same_root_cap = 2          # Broad query with dominant root: enforce diversity
else:
    same_root_cap = 3          # Default
```

### CE top-3 protection

```python
# After structure rerank, ensure CE top-3 are not demoted below rank 5
for doc in ce_top3:
    if doc not in final_top5 and not severe_diversity_violation:
        swap into top5, demoting lowest-ranked non-protected doc
```

`severe_diversity_violation`: all top-5 from same root (override protection in this case).

### Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| `L3_ROOT_WEIGHT` | 0.15 | Down from 0.30 |
| `L3_SAME_ROOT_CAP_DEFAULT` | 3 | Default per-root cap |
| `L3_SAME_ROOT_CAP_SCOPE_QUERY` | 5 | Scope query cap |
| `L3_SAME_ROOT_CAP_BROAD_QUERY` | 2 | Broad query cap |
| `L3_PROTECT_CE_TOP3` | true | Don't demote CE top-3 below rank 5 |

---

## New Variant Registration

Register `V3Q_LAYERED` in `scripts/rag_eval/variants.py`:

```python
"V3Q_LAYERED": {
    "env": {
        "RAG_INDEX_PROFILE": "v3_quality",
        "TEXT_MODE": "title_context_filename",
        "LAYERED_RERANK_ENABLED": "true",
        "RAG_CANDIDATE_K": "80",
        "L0_DENSE_TOP_K": "80",
        "L0_SPARSE_TOP_K": "80",
        "L0_HYBRID_GUARANTEE_K": "20",
        "L1_TOP_FILES": "12",
        "L1_MAX_CANDIDATES": "40",
        "L2_CE_DEFAULT_K": "32",
        "L2_CE_TOP_N": "15",
        "L3_ROOT_WEIGHT": "0.15",
        "L3_SAME_ROOT_CAP_DEFAULT": "3",
        "MILVUS_RRF_K": "80",
        "MILVUS_SPARSE_DROP_RATIO": "0.1",
        "RERANK_SCORE_FUSION_ENABLED": "true",
        "RERANK_PAIR_ENRICHMENT_ENABLED": "true",
        "STRUCTURE_RERANK_ENABLED": "true",
        "SAME_ROOT_CAP": "3",
    },
    "index_profile": "v3_quality",
    "collection": "embeddings_collection_v3_quality",
    "text_mode": "title_context_filename",
    "skip_reindex": False,
}
```

---

## Trace Metrics

### L0 metrics

| Metric | Description |
|--------|-------------|
| `l0_dense_count` | Candidates from dense path |
| `l0_sparse_count` | Candidates from sparse path |
| `l0_hybrid_count` | Candidates from hybrid guarantee |
| `l0_pool_size` | Total unique candidates after merge |
| `l0_fallback_used` | Whether hybrid fallback triggered |
| `l0_union_file_recall` | File recall in L0 pool |
| `l0_union_root_recall` | Root recall in L0 pool |

### L1 metrics

| Metric | Description |
|--------|-------------|
| `l1_pool_before_filter` | L0 pool size entering L1 |
| `l1_selected_count` | Candidates after L1 |
| `l1_slot_a_count` | File-aware slot contribution |
| `l1_slot_b_count` | Route guarantee slot contribution |
| `l1_slot_c_count` | Guarantee slot contribution |
| `l1_file_recall` | File recall after L1 |
| `l1_chunk_recall` | Chunk recall after L1 |
| `l1_root_recall` | Root recall after L1 |
| `l1_page_recall` | Page recall after L1 |
| `l1_file_page_recall` | File+Page recall after L1 |
| `l1_correct_file_dropped` | Whether correct file was dropped |
| `l1_correct_chunk_dropped` | Whether correct chunk was dropped |
| `l1_correct_root_dropped` | Whether correct root was dropped |
| `l1_drop_reason` | Enum: `not_in_l0_dense`, `not_in_l0_sparse`, `not_in_l0_union`, `dropped_by_file_topk`, `dropped_by_per_file_cap`, `dropped_by_slot_quota`, `dropped_by_l1_max_candidates`, `dropped_by_dedup` |

### L2 metrics

| Metric | Description |
|--------|-------------|
| `ce_input_k` | Actual CE input count |
| `ce_latency_ms` | CE inference time |
| `ce_score_margin` | Top-1 vs top-2 CE score gap |
| `ce_k_high_conf_ratio` | Fraction of queries using K=25 |
| `ce_k_default_ratio` | Fraction using K=32 |
| `ce_k_low_conf_ratio` | Fraction using K=40 |

### L3 metrics

| Metric | Description |
|--------|-------------|
| `structure_gain_rate` | Fraction where structure improved ranking |
| `structure_drop_rate` | Fraction where structure hurt ranking |
| `structure_tiebreak_count` | Times structure only affected tied scores |
| `ce_top3_demoted` | Whether CE top-3 was protected |

---

## Latency Estimate

| Stage | V3Q baseline | V3Q_LAYERED estimate |
|-------|-------------|---------------------|
| Embedding | ~50ms | ~50ms |
| Milvus | ~40ms | ~60ms (parallel dense+sparse+hybrid) |
| L1 prefilter | — | ~3ms |
| CrossEncoder | **4,300ms** (80 candidates) | **1,300-1,700ms** (25-40 candidates) |
| Structure rerank | ~0.2ms | ~0.2ms |
| **Total P50** | **~4,500ms** | **~1,500-1,800ms** |

Latency depends on adaptive K distribution. Target: 50%+ queries at K=25/32, 15% at K=40.

---

## Quality Targets

| Metric | Constraint |
|--------|-----------|
| File@5 | Degradation <= 0.3pp (from 0.955) |
| File+Page@5 | Degradation <= 0.5pp (from 0.727) |
| Chunk@5 | No degradation vs V3Q_OPT |
| L1 File Recall@40 | >= 0.99 |
| L1 Chunk Recall@40 | >= current K=50 baseline |
| L1 Root Recall@40 | >= 0.98 |
| P50 latency | >= 50% reduction |
| P95 latency | No regression |
| structure_drop_rate | Significantly below current 10.4% |

Core quality gates (must pass both):
1. File@5 degradation <= 0.3pp
2. Chunk@5 not below V3Q_OPT

---

## Experiment Matrix

| ID | L0 | L1 | L2 K | L3 | Purpose |
|----|-----|-----|------|-----|---------|
| C0 | Current hybrid | None | 80 | Current | V3Q baseline |
| C1 | Current hybrid | None | 50 | Current | V3Q_OPT baseline |
| C2 | Split 80/80 | Simple fusion top40 | 40 | Current | Validate L0 split |
| C2.5 | Split 80/80 + hybrid top20 | Simple fusion top40 | 40 | Current | Validate hybrid guarantee |
| C3 | Split 80/80 | File-aware | 40 | Current | Validate L1 |
| C4 | Split 80/80 | File-aware | 32 | Current | Compress CE input |
| C5 | Split 80/80 | File-aware | Adaptive 25-40 | Current | Validate adaptive K |
| **C6** | **Split 80/80 + hybrid top20** | **File-aware + quotas** | **Adaptive 25-40** | **Weak** | **Recommended candidate** |
| C6.5 | Split 80/80 + hybrid top20 | File-aware + quotas | Adaptive 25-40 | Weak + CE top20 | Validate wider L2 output |
| C7 | Split 60/60 | File-aware + quotas | Adaptive 25-40 | Weak | Validate L0 compression |

Recommended path: C0 → C2 → C3 → C6 → production.

---

## File Changes

### New files

- `backend/rag/layered_rerank.py` — L0 split_retrieve, L1 prefilter, L2 adaptive K logic
- `backend/rag/l1_scoring.py` — Score normalization, file aggregation, slot construction

### Modified files

- `backend/infra/vector_store/milvus_client.py` — Add `split_retrieve` method
- `backend/rag/utils.py` — Wire LAYERED_RERANK_ENABLED into pipeline
- `backend/rag/context.py` — L3 weak structure + CE top3 protection
- `backend/rag/trace.py` — New L0/L1/L2/L3 trace fields
- `scripts/rag_eval/variants.py` — Register V3Q_LAYERED variant
- `scripts/evaluate_rag_matrix.py` — New trace metrics in eval output

### Config (environment variables)

All configuration via environment variables with `LAYERED_RERANK_ENABLED` as the master toggle. When disabled, pipeline falls back to current V3Q behavior exactly.
