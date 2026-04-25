# RAG v4 Execution Design

Date: 2026-04-25
Status: approved approach, implementation pending
Approach: converge the existing uncommitted v4 implementation

## Context

The workspace already contains substantial uncommitted RAG v4 work. Notable files include:

- `backend/query_plan.py`
- `scripts/analyze_rag_misses.py`
- `scripts/derive_natural_query_subset.py`
- `tests/test_query_plan_parser.py`
- `tests/test_scoped_global_rrf.py`
- `tests/test_rerank_pair_enrichment.py`
- `tests/test_bm25_state_isolation.py`
- large edits in `backend/rag_utils.py`, `backend/rag_pipeline.py`, `backend/rag_diagnostics.py`, and `scripts/evaluate_rag_matrix.py`

The implementation plan must treat these as user-owned work in progress. The goal is not to restart from the last clean commit. The goal is to audit, repair, complete, and verify the current v4 implementation while preserving existing uncommitted changes.

## Primary Goal

Ship the RAG v4 retrieval and evaluation reform in controlled phases:

1. Stabilize fallback behavior and evaluation metrics.
2. Add query planning, document scoping, scoped/global hybrid retrieval, and metadata-aware reranking.
3. Improve diagnostics and test coverage.
4. Document Phase E answer evaluation as a roadmap item.
5. Defer retrieval-text reindex until the no-reindex stages pass their gates.

## Non-Goals

- Do not reset, revert, or overwrite unrelated uncommitted changes.
- Do not perform Phase C3 full reindex before S2HR meets its acceptance gate.
- Do not introduce `MILVUS_COLLECTION_NAME`; continue using `MILVUS_COLLECTION`.
- Do not make UI redesign changes.
- Do not add new dependencies unless already introduced by the current work in progress and required to make tests pass.
- Do not enable fallback by default.

## Architecture

The existing pipeline remains the boundary:

`DocumentLoader -> MilvusWriter -> MilvusManager -> retrieve_documents -> rag_pipeline -> evaluation scripts`

The v4 changes add narrowly scoped modules and helpers:

- `backend/query_plan.py` parses raw user questions into `semantic_query`, document hints, matched files, route, anchors, and model numbers.
- `backend/rag_utils.py` remains the single retrieval entry point. It applies query planning, scoped/global hybrid retrieval, heading lexical scoring, rerank pair enrichment, rerank cache keys, and trace fields.
- `backend/rag_pipeline.py` owns graph-level fallback execution and trace semantics.
- `backend/rag_diagnostics.py` and `scripts/analyze_rag_misses.py` classify miss causes using the v4 diagnostic taxonomy.
- `scripts/evaluate_rag_matrix.py` owns variant definitions, metrics, summary markdown, paired comparisons, and dataset profile execution.
- `backend/document_loader.py` owns retrieval-text formatting for the later reindex stage.
- `backend/embedding.py` owns BM25 state path isolation.

## Data Flow

1. `retrieve_documents(raw_query, top_k, context_files)` receives the original query.
2. `parse_query_plan()` produces:
   - `raw_query`: unchanged user input.
   - `clean_query`: display-oriented query after removing book-title wrapper text.
   - `semantic_query`: retrieval query with document wrappers removed and model numbers conditionally removed only when document scope is confident.
   - `matched_files`: filename matches with scores.
   - `scope_mode`: `filter`, `boost`, or `none`.
   - `route`: `scoped_hybrid` or `global_hybrid`.
3. Dense and sparse embeddings are computed once from `semantic_query`.
4. If route is scoped:
   - scoped Milvus hybrid retrieval runs with a filename filter.
   - global Milvus hybrid retrieval runs without filename filter.
   - results are merged by weighted RRF, default 80/20.
5. If route is global:
   - standard global hybrid retrieval runs with the leaf-level filter.
6. Optional heading lexical scoring reorders candidates before rerank when scope and heading hints are present.
7. Optional rerank pair enrichment builds pair text from filename, section path, page, anchor, heading, and body.
8. Rerank cache signatures hash the final pair text, not just raw retrieval text.
9. Structure rerank and confidence evaluation run after rerank.
10. `rag_pipeline` records graph trace fields:
    - `fallback_required_raw`
    - `fallback_executed`
    - `fallback_disabled`
    - `graph_path`

## Task Board

### Phase 0 - Evaluation Reform

Status: pending audit and completion

- Keep historical baselines stable.
- Add or verify `B0_legacy`, `S1`, `S2`, `S2H`, `S2HR`, and `S3` variants.
- Make primary metrics:
  - File@5
  - File+Page@5
  - CandidateRecallBeforeRerank
  - HardNeg@5
  - Anchor@5
  - MRR
  - P50/P95 latency
- Render Chunk@5 and Root@5 as `n/a (qrels missing)` in main summary, paired comparisons, and diagnostics.
- Add open-retrieval-natural dataset support.

### Phase A - Fallback Stop-Bleed

Status: pending audit and completion

- Set `CONFIDENCE_GATE_ENABLED=false` by default.
- Set `RAG_FALLBACK_ENABLED=false` by default.
- Short-circuit `grade_documents_node` when fallback is disabled.
- Preserve trace distinction between required, executed, and disabled fallback.
- Record `graph_path=linear_initial_only` for disabled fallback.
- Route fallback LLM work to `FAST_MODEL` when manually enabled.
- Update `.env.example`.

### Phase B - Miss Diagnostics

Status: pending audit and completion

- Replace legacy categories with:
  - `file_recall_miss`
  - `page_miss`
  - `ranking_miss`
  - `hard_negative_confusion`
  - `low_confidence`
- Add miss analysis output for candidate recall buckets, filename fuzzy match distribution, hard-negative families, anchor hit rate, rerank drops, and top wrong files.
- Update `docs/rag_evaluation.md`.

### Phase C0/C1 - QueryPlan and Document Scope

Status: pending audit and completion

- Validate `QueryPlan` schema and parser.
- Validate filename normalization and compound matching.
- Build lazy filename registry keyed by `MILVUS_COLLECTION` and `milvus_index_version`.
- Ensure user `context_files` is a hard boundary and can only be narrowed.
- Implement route rules:
  - `filter` and `boost` route to scoped hybrid.
  - `none` routes to global hybrid.
  - weak matches enter trace only.
- Implement scoped/global parallel retrieval with `ThreadPoolExecutor`.
- Ensure dense and sparse embeddings are computed once and reused.
- Merge scoped/global candidates by weighted RRF and dedupe by `chunk_id`.

### Phase C2/C2R - Ranking Quality

Status: pending audit and completion

- Apply heading lexical scoring only when scoped and heading hints exist.
- Start with `HEADING_LEXICAL_WEIGHT=0.20`.
- Add metadata-aware rerank pair enrichment before reindex.
- Hash final pair text in `_rerank_doc_signatures`.
- Verify enrichment toggles change cache keys.

### Phase C3 - Reindex Preparation

Status: deferred until S2HR gate passes

- Add retrieval-text mode `title_context_filename`.
- Keep text length within Milvus field limits.
- Isolate BM25 state by `MILVUS_COLLECTION` and `EVAL_RETRIEVAL_TEXT_MODE`.
- Document double-collection migration through `MILVUS_COLLECTION`.
- Do not execute full reindex in this implementation batch unless explicitly requested after gates pass.

### Phase D - Scenario Evaluation

Status: pending after implementation tests pass

- Run smoke first.
- Run frozen dataset.
- Run gold and open-retrieval-natural when local services are available.
- Compare each phase against its gate.

### Phase E - Answer Evaluation Roadmap

Status: documentation only

- Define faithfulness, answer relevance, and citation coverage.
- Do not add Phase E runtime implementation in this batch.

## Acceptance Gates

### S1

- `fallback_executed=0`
- P50 returns close to the current linear baseline
- no graph fallback by default

### S2

- File@5 >= 0.40 on gold
- CandidateRecallBeforeRerank >= 0.65 on gold
- P50 <= 1500 ms

### S2H

- File@5 >= 0.50 on gold
- File+Page@5 >= 0.30 on gold

### S2HR

- File@5 >= 0.55 on gold
- File+Page@5 >= 0.35 on gold
- HardNeg@5 <= 0.18 on gold

### S3

- File@5 >= 0.70 qualified, >= 0.80 ideal
- File+Page@5 >= 0.55 qualified, >= 0.70 ideal
- P50 <= 1500 ms
- File@5 >= 0.50 on open-retrieval-natural

## Error Handling

- If filename registry lookup fails, fall back to global hybrid.
- If scoped retrieval fails, preserve global retrieval as fallback where possible.
- If parallel Milvus retrieval proves unstable, degrade to serial scoped then global queries.
- If reranker fails, preserve existing ranked candidates and record stage error.
- If pair enrichment creates unexpected empty pair text, fall back to `_doc_retrieval_text`.
- If evaluation qrels lack chunk/root identifiers, render `n/a` rather than false zeroes.

## Testing Plan

Run focused tests first:

- `tests/test_filename_normalization.py`
- `tests/test_filename_match_score.py`
- `tests/test_query_plan_parser.py`
- `tests/test_document_scope_matching.py`
- `tests/test_scoped_global_rrf.py`
- `tests/test_heading_lexical_scoring.py`
- `tests/test_rerank_pair_enrichment.py`
- `tests/test_bm25_state_isolation.py`
- `tests/test_fallback_disabled_routing.py`
- `tests/test_diagnostics_v4.py`
- `tests/test_evaluate_rag_matrix.py`
- `tests/test_rag_utils.py`
- `tests/test_rag_pipeline.py`
- `tests/test_rag_pipeline_fast_path.py`
- `tests/test_rag_observability.py`

Then run smoke evaluation:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile smoke --variants B0,S1,S2 --skip-reindex --limit 10 --run-id rag-v4-smoke
```

Then run frozen evaluation:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile frozen --variants B0,S1,S2,S2H,S2HR --skip-reindex --run-id rag-v4-frozen
```

Gold and natural dataset runs are the final verification when Milvus, Redis, local embedding, and rerank services are available.

## Implementation Policy

- Inspect before editing.
- Use the existing v4 files as the source of truth unless tests show they are wrong.
- Prefer small fixes over rewrites.
- Keep public function names stable where tests or runtime code already refer to them.
- Commit only intentionally staged files.
- Final implementation report must include changed files, simplifications made, verification evidence, and remaining risks.

