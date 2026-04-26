# RAG Eval Data Closure & Rerank Optimization Design

Date: 2026-04-25
Status: Draft

## Problem Statement

The RAG evaluation report (`rag-v4-full-gold-rerun`) shows `File@5=0.056~0.072`, which appears to indicate poor retrieval/reranking performance. However, the root cause is **not** the reranker — it is a fundamental data closure failure:

- **Milvus `embeddings_collection`** contains only **5 documents** (1325 leaf chunks).
- **Gold dataset** requires **60 unique expected files** across 125 samples.
- Only **11/125 samples (8.8%)** have their expected file in the current index.
- The `File@5 ≈ 0.072` ceiling is exactly explained by this coverage gap.

Additionally, `CandidateRecall=0.744~0.768` is misleading: it counts keyword/page/anchor weak hits, not strict "expected file in candidate pool" recall.

The `C:\Users\goahe\Desktop\Project\doc` directory contains all 60 expected files (100% coverage), but has never been indexed into Milvus.

All prior variant comparisons (B0/S1/S2/S2H/S2HR/S3) were conducted on a **broken index** and their conclusions are invalid.

## Key Changes

### Phase 1: Fix Evaluation Data Closure

#### 1.1 Reindex with Full Document Set

Use `C:\Users\goahe\Desktop\Project\doc` as the evaluation document source (not copied into the repo).

**Two collections to build:**

| Collection | Text Mode | Purpose |
|---|---|---|
| `embeddings_collection` | `title_context` | B0_legacy, S1, S2, S2H, S2HR |
| `embeddings_collection_v2` | `title_context_filename` | S3 |

**Reindex commands:**

```bash
# Collection 1: title_context mode
uv run python scripts/reindex_knowledge_base.py \
  --documents-dir "C:\Users\goahe\Desktop\Project\doc" \
  --collection embeddings_collection \
  --text-mode title_context

# Collection 2: title_context_filename mode
uv run python scripts/reindex_knowledge_base.py \
  --documents-dir "C:\Users\goahe\Desktop\Project\doc" \
  --collection embeddings_collection_v2 \
  --text-mode title_context_filename
```

**Verification:**
- After reindex, query Milvus for unique filenames at leaf level.
- Assert: all 60 expected files from gold dataset are present.
- Run reindex dry-run first to confirm chunk counts.

#### 1.2 Add Preflight Check to evaluate_rag_matrix.py

Before running any evaluation, validate that the gold dataset's `expected_files` exist in the Milvus collection. If coverage < 95%, **fail with an explicit error** instead of producing a misleading report.

Implementation:
1. Load gold dataset, extract all `expected_files`.
2. Query Milvus for distinct filenames in the target collection.
3. Compare and report coverage.
4. If coverage < 95%, exit with error unless `--skip-coverage-check` is explicitly set.

### Phase 2: Fix Metrics

#### 2.1 Add File-Level Candidate Metrics

Current `CandidateRecall` is a weak metric. Add:

| New Metric | Definition |
|---|---|
| `file_candidate_recall_before_rerank` | % of samples where expected file appears in the pre-rerank candidate pool (strict, filename-level) |
| `file_rerank_drop_rate` | % of samples where expected file was in candidates but dropped after rerank |
| `file_rank_before_rerank_p50/p95` | Rank of expected file in pre-rerank candidates (when present) |

These metrics directly answer: "Is the problem in recall or in ranking?"

#### 2.2 Clarify Existing CandidateRecall

Rename or split the current `CandidateRecall`:

- `weak_any_candidate_recall` — current metric (includes keyword/anchor/page weak matches)
- `file_candidate_recall` — strict: expected file's chunks appear in candidate list

#### 2.3 Fill Missing Qrels

`Chunk@5` and `Root@5` show `n/a (qrels missing)`. The gold dataset already has `gold_chunk_ids` and `expected_root_ids` fields. Wire these up in the evaluation script so chunk-level and root-level metrics are computed.

### Phase 3: Rerank Optimization (Only After Phase 1+2)

**Do not tune the reranker until the index is complete and metrics are trustworthy.**

Once Phase 1+2 are done, the optimization direction is:

#### 3.1 Expand Rerank Window

| Parameter | Current | Proposed | Rationale |
|---|---|---|---|
| `RAG_CANDIDATE_K` | 0 (=50) | 80~120 | More candidates from Milvus |
| `RERANK_INPUT_K_GPU` | 20 | 50~80 | Let reranker see all candidates |
| `RERANK_TOP_N` | 0 (=5) | 20~30 | Larger output window for ablation |

Ablation plan: try 80/50/20 first, then 120/80/30.

#### 3.2 Score Blending

Current: reranker score completely replaces RRF score.
Proposed: `final_score = α * rerank_norm + (1-α) * rrf_norm`, with `α=0.7` default.

New env var: `RERANK_BLEND_ALPHA=0.7`

Normalization:
- RRF: min-max normalization across candidates
- Rerank: sigmoid normalization (scores can be negative)

#### 3.3 Lower QueryPlan Thresholds

| Parameter | Current | Proposed | Rationale |
|---|---|---|---|
| `DOC_SCOPE_MATCH_FILTER` | 0.85 | 0.50 | Fuzzy match histogram shows most matches in 0.3-0.7 range |
| `DOC_SCOPE_MATCH_BOOST` | 0.60 | 0.35 | Trigger scoped retrieval for more queries |

#### 3.4 Force Rerank Pair Enrichment

Set `RERANK_PAIR_ENRICHMENT_ENABLED=true` by default. The enriched format `[filename][section_path][p.page][anchor] body` gives the reranker access to product model information.

#### 3.5 SAME_ROOT_CAP Adjustment

From 2 to 3. When a query explicitly targets one document, multiple chunks from that document should be allowed.

### Phase 4: Confidence Gate + Fallback (Deferred)

Enable `CONFIDENCE_GATE_ENABLED=true` with fallback to LLM query rewrite. This is deferred to after Phase 3 because we need trustworthy baseline metrics first.

## Execution Plan

### Step 1: Reindex (highest priority)

1. Dry-run reindex to confirm document count, chunk counts, retrieval_text lengths.
2. Build `embeddings_collection` with `title_context` mode from `C:\Users\goahe\Desktop\Project\doc`.
3. Build `embeddings_collection_v2` with `title_context_filename` mode.
4. Verify both collections contain all 60 expected filenames.

### Step 2: Re-evaluate baseline

1. Run `B0_legacy, S1_linear, S2` on the new index.
2. Verify `File@5` is significantly higher than 0.072.
3. If not, investigate ingestion/metadata issues.

### Step 3: Add preflight + metrics

1. Add coverage preflight check to `evaluate_rag_matrix.py`.
2. Add `file_candidate_recall_before_rerank`, `file_rerank_drop_rate`, `file_rank_before_rerank_p50/p95`.
3. Wire up `gold_chunk_ids` for `Chunk@5`/`Root@5`.

### Step 4: Rerank ablation

1. Expand rerank window parameters.
2. Implement score blending.
3. Lower QueryPlan thresholds.
4. Run ablation with S2H/S2HR/S3 variants.

### Step 5: Answer-eval

Only after retrieval metrics are satisfactory. Run `--mode answer-eval` for end-to-end quality.

## Test Plan

| Checkpoint | Verification |
|---|---|
| After reindex | Milvus unique filenames count = 60 |
| After reindex | Gold expected file coverage = 60/60 |
| After baseline eval | File@5 > 0.30 (conservative estimate) |
| After metrics fix | file_candidate_recall and file_rerank_drop_rate are computed |
| After rerank ablation | Identify which variant + params yield best File@5 |
| After answer-eval | Faithfulness, answer relevance, citation coverage reported |

## Assumptions

- Use `C:\Users\goahe\Desktop\Project\doc` as the gold evaluation document source. Files are not copied into the repo.
- No new dependencies. Reuse existing `reindex_knowledge_base.py`, `DocumentLoader`, Milvus, BM25, QueryPlan.
- Phase 1 goal: make retrieval evaluation trustworthy. Reranker model replacement, GraphRAG, LLM fallback are Phase 2+.
- S3 variant results from the old index are invalid because `skip-reindex=true` was used but S3 requires `title_context_filename` reindex.
