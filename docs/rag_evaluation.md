# SuperHermes RAG Evaluation Runbook

This runbook keeps the current RAG architecture intact and focuses on controlled evaluation.

## Evaluation Policy

1. Fix the retrieval evaluation baseline first.
2. Tune retrieval knobs before changing embeddings, vector database, or chunking.
3. Run a chunk profile experiment only when miss analysis shows chunk-boundary failures.
4. Defer faithfulness, citation support, and abstention judging until retrieval quality is stable.

## Gate 0 Baseline - 2026-04-24

Purpose: record the current runnable state before RAG performance changes.

- Runtime evidence: `.jbeval/logs/uvicorn-8000.stderr.log` records `Application startup complete` and `Uvicorn running on http://127.0.0.1:8000`.
- Embedding: `EMBEDDING_PROVIDER=local`, `EMBEDDING_MODEL=BAAI/bge-m3`, `EMBEDDING_DEVICE=auto`, `DENSE_EMBEDDING_DIM=1024`.
- Reranker: `RERANK_PROVIDER=local`, `RERANK_MODEL=BAAI/bge-reranker-v2-m3`, `RERANK_DEVICE=auto`.
- CUDA check: `cuda_available=True`, `cuda_device_count=1`.
- Milvus collection: `MILVUS_COLLECTION=embeddings_collection`.
- Storage: `DATABASE_URL=postgresql+psycopg2://postgres:postgres@127.0.0.1:5433/langchain_app`, `REDIS_URL=redis://127.0.0.1:6379/0`.
- Targeted pre-change tests: `uv run pytest tests/test_rag_utils.py tests/test_rag_pipeline_fast_path.py tests/test_milvus_client.py` passed with `12 passed`.
- Existing smoke report: `.jbeval/reports/rag-phase0-gpu-smoke/summary.md`, B0 `Rows=1`, `Hit@5=1.000`, `P50=61153.454 ms`.
- Existing graph report: `.jbeval/reports/rag-phase25-graph-gpu-local/summary.md`, B0 `P50=4034.307 ms`, F1 `P50=88023.516 ms`, F1 fallback rate `0.909`.
- Existing gold report: `.jbeval/reports/rag-phase2-gold-confirm-local/summary.md`, B0 `Rows=125`, `Hit@5=0.352`, `P50=4326.166 ms`, `P95=5156.644 ms`.

## Phase 1 Observability Baseline - 2026-04-24

Behavior policy: only trace fields were added; retrieval, reranking, fallback, and answer generation decisions are unchanged.

- Added backend timing fields under `rag_trace.timings`: `embed_dense_ms`, `embed_sparse_ms`, `milvus_hybrid_ms`, `milvus_dense_fallback_ms`, `rerank_ms`, `structure_rerank_ms`, `confidence_ms`, `grader_ms`, `rewrite_router_ms`, `stepback_llm_ms`, `hyde_llm_ms`, `expanded_retrieve_ms`, `total_retrieve_ms`, `total_rag_graph_ms`, `final_llm_ms`.
- Added backend degradation trace field: `rag_trace.stage_errors`, preserving top-level `retrieval_mode` for the high-level mode.
- Added context size fields: `context_chars`, `retrieved_chunk_count`, `final_context_chunk_count`.
- Added frontend display for timing rows and degradation events in the existing RAG trace panel.
- Parent chunk level distribution query result: `chunk_level=1,count=1272`; no L2 rows were present in `parent_chunks`.
- Phase 1 targeted tests: `uv run pytest tests/test_rag_observability.py tests/test_rag_utils.py tests/test_rag_pipeline_fast_path.py tests/test_milvus_client.py tests/test_tools_request_context.py` passed with `20 passed`.

## Phase 2A/2B Smoke - 2026-04-24

Changes under test:

- `RERANK_INPUT_K_CPU=10` and `RERANK_INPUT_K_GPU=20` cap reranker inputs before local `predict()` or API/Ollama `post()`.
- Rerank trace now includes `rerank_input_count`, `rerank_output_count`, `rerank_input_cap`, and `rerank_input_device_tier`.
- `step_back_expand()` now uses one structured JSON LLM call instead of separate question-generation and answer-generation calls; invalid JSON falls back to the original query.

Validation:

- Targeted regression: `uv run pytest tests/test_rag_observability.py tests/test_rag_utils.py tests/test_rag_pipeline_fast_path.py tests/test_milvus_client.py tests/test_tools_request_context.py` passed with `24 passed`.
- Frontend syntax: `node --check frontend/script.js` passed.
- Diff hygiene: `git diff --check ...` completed with no whitespace errors.
- Smoke benchmark: `uv run python scripts\evaluate_rag_matrix.py --dataset-profile frozen --variants B0 --skip-reindex --limit 20 --run-id rag-phase2ab-smoke`.
- Smoke result: `.jbeval/reports/rag-phase2ab-smoke/summary.md`, B0 `Rows=20`, `Hit@5=0.350`, `P50=835.791 ms`, `P95=2411.931 ms`, `Error=0.000`.

## Phase 2 Conditional Decisions - 2026-04-24

Decision data:

- Trace smoke: `.jbeval/reports/rag-phase2ab-trace-smoke`, B0 `Rows=5`, `Hit@5=0.800`, `P50=741.847 ms`, `P95=18006.991 ms`.
- Hot-path average after the first cold sample: dense embedding about `39 ms`, Milvus hybrid about `17 ms`, rerank about `700 ms`.
- Graph context smoke: `.jbeval/reports/rag-phase2ab-graph-context-smoke`, `context_chars` ranged from about `1686` to `2653` with `final_context_chunk_count=5`.
- Phase 2D trace check: `.jbeval/reports/rag-phase2d-trace-check`, `rerank_input_count=20`, `rerank_input_cap=20`, `rerank_cache_enabled=True`, `context_chars=2368..2653`.

Decisions:

- 2C query embedding cache: not implemented now. Hot dense embedding is about 5% of retrieval latency, below the 15% trigger.
- 2D retrieval/rerank cache: implemented for rerank only. Rerank dominates hot-path latency, while Milvus hybrid is about `17 ms`; full retrieval-result caching is deferred to avoid broad invalidation complexity for a small measured gain.
- 2E context budget: not implemented now. Current graph fast path is below the `12000` character trigger. This should be revisited for attachment-heavy turns or fallback-expanded contexts.

Implementation notes:

- Rerank cache key binds query, rerank provider/model/host/device tier, ordered chunk IDs, retrieval text hashes, BM25 `_total_docs`, and `milvus_index_version`.
- `milvus_index_version` is incremented after Milvus writes, deletes, and collection drops. Redis failures degrade to no cache.
- Targeted regression: `uv run pytest tests/test_rag_observability.py tests/test_rag_utils.py tests/test_milvus_index_version.py tests/test_evaluate_rag_matrix.py` passed with `35 passed`.

## Phase 3A/3B Structural Notes - 2026-04-24

Phase 3A fallback controls:

- Added `RAG_FALLBACK_TIMEOUT_SECONDS=15`.
- Fallback deadline starts in `rewrite_question_node` and is carried into `retrieve_expanded`.
- Complex fallback retrieves HyDE and step-back branches in parallel through a bounded thread pool.
- If router, expansion LLM, or expanded retrieval exceeds the deadline, the graph returns the initial retrieval result and marks `fallback_timed_out=True`, `fallback_returned_initial=True`.
- Targeted regression: `uv run pytest tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py` passed with `9 passed`.

Phase 3B Milvus client reuse decision:

- Current local PyMilvus version is `2.6.10`.
- Official PyMilvus v2.6 docs describe `MilvusClient` connection usage but do not document a thread-safety guarantee for sharing one `MilvusClient` across request threads.
- `AsyncMilvusClient` is documented as early-stage and not advised for production use, so it is not a suitable replacement here.
- The project code already contains a design note to avoid sharing `MilvusClient` and to create a fresh client per operation attempt to avoid cross-thread/cross-request invalidation.
- Decision: do not implement thread-local or shared Milvus client reuse in this phase. Keep the existing fresh-client strategy and rely on Phase 1 timings, which showed hot-path Milvus hybrid search around `17 ms`, far below rerank latency.

Phase 3C/3D service-path changes:

- Conversation history summarization now uses `FAST_MODEL` when configured; missing or failed fast-model initialization falls back to the main `MODEL`.
- `/documents/upload` now returns a pending upload task instead of blocking on parsing, embedding, and Milvus writes.
- New status API: `GET /documents/status/{task_id}` returns `pending|processing|done|failed`, processed chunk count, message, and error.
- The frontend upload flow polls status every 2 seconds and only adds the uploaded file to chat context after the indexing task reaches `done`.
- Targeted regression: `uv run pytest tests/test_document_upload_tasks.py tests/test_agent_summary_model.py tests/test_tools_request_context.py tests/test_rag_pipeline.py tests/test_rag_pipeline_fast_path.py tests/test_rag_observability.py` passed with `21 passed`; `node --check frontend/script.js` passed.

## Phase 4 Quality Decisions - 2026-04-24

Evidence reviewed:

- Phase 1 parent chunk distribution showed only `chunk_level=1`, `count=1272`; no real L2 parent chunks exist in the current index.
- Recent small trace checks had no misses in `.jbeval/reports/rag-phase2d-trace-check` and `.jbeval/reports/rag-phase2ab-graph-context-smoke` is too small to justify chunking changes.
- The graph comparison `.jbeval/reports/rag-phase25-graph-gpu-local/summary.md` shows `F1` fallback hurt more than helped overall (`helped=0.045`, `hurt=0.114`, `P50=88023.516 ms`), but HyDE has only one `complex` sample there, so this is not enough evidence to disable HyDE specifically.

Decisions:

- Chunk size and overlap are not changed in this phase. Current misses are not clearly attributable to chunk boundary failures, and changing chunking would require reindexing plus a full `.jbeval` comparison.
- HyDE is not disabled in this phase. Existing evidence points to confidence/fallback policy cost, not a HyDE-specific regression.
- Auto-merge is simplified from the previous fake two-step path to a real single `L3->L1` path. This removes an ineffective `L3->L2->L1` assumption while preserving the current indexed hierarchy.

Implementation notes:

- `_auto_merge_documents` now records `auto_merge_path="L3->L1"` when enabled.
- Targeted regression: `uv run pytest tests/test_rag_utils.py` passed with `14 passed`.

## Dataset Profiles

- `frozen`: `.jbeval/datasets/rag_doc_frozen_eval_v1.jsonl`, default quick-regression benchmark (44 samples).
- `gold`: `.jbeval/datasets/rag_doc_gold.jsonl`, full benchmark with `《》` book-title hints (125 samples).
- `natural`: `.jbeval/datasets/rag_doc_gold_natural_v1.jsonl`, open-retrieval-natural benchmark without `《》` hints (125 samples).
- `smoke`: frozen dataset with an implicit 10-row limit unless `--limit` is passed.
- `custom`: explicit `--dataset <path>`.

Every run validates the dataset schema before retrieval starts and writes dataset hash/config metadata to `config.json`.

## Primary Metrics

| Metric | Definition | Notes |
| --- | --- | --- |
| `File@5` | top5 contains correct filename | Main dashboard metric |
| `File+Page@5` | top5 contains correct filename AND correct page | Strict match |
| `CandidateRecallBeforeRerank` | correct file present in pre-rerank candidates | Measures raw recall |
| `HardNeg@5` | fraction of top5 from hard-negative files | **Gold only**; `-` on natural/frozen |
| `Anchor@5` | top5 contains expected anchor | Structural match |
| `MRR` | reciprocal rank of first correct file | Ranking quality |
| `P50 / P95` | retrieval latency percentiles | Performance budget |

**Deprecated**: `Chunk@5` and `Root@5` show `n/a (qrels missing)` because gold_chunk_ids are not available in the current dataset. This marker must appear in the main dashboard, paired comparisons, and diagnostics sections so reports do not silently mix old qrels-based metrics with file/page scoring.

## Staged Acceptance Thresholds

### S2 (QueryPlan + doc scope 80/20 parallel, no reindex)
- `File@5 >= 0.40`
- `CandidateRecall >= 0.65`
- `P50 <= 1500ms`
- Not met -> fix QueryPlan parser / filename registry / matcher

### S2H (S2 + heading lexical scoring)
- `File@5 >= 0.50`
- `File+Page@5 >= 0.30`
- Not met -> adjust `HEADING_LEXICAL_WEIGHT` or revert to S2

### S2HR (S2H + metadata-aware rerank pair enrichment)
- `File@5 >= 0.55`
- `File+Page@5 >= 0.35`
- `HardNeg@5 <= 0.18`
- Not met -> check pair enrichment format or rerank model capability

### S3 (full: reindex title_context_filename + all enrichments)
- `File@5 >= 0.70` (pass) / 0.80 (ideal)
- `File+Page@5 >= 0.55` (pass) / 0.70 (ideal)
- `P50 <= 1500ms`
- `open-retrieval-natural` `File@5 >= 0.50`

### Fallback re-enable threshold
- `helped > hurt` AND P50 increment < 1000ms

## Variant Definitions (v4)

| Variant | Description | Key env |
| --- | --- | --- |
| `B0_legacy` | Current production configuration, no evaluation env overrides | none |
| `B0` | Compatibility baseline from earlier matrices | `RERANK_TOP_N=0` |
| `S1_linear` | Linear path: gate off + fallback off | `RAG_FALLBACK_ENABLED=false` |
| `S2` | + QueryPlan + doc scope 80/20 parallel | `DOC_SCOPE_MATCH_FILTER=0.85`, `DOC_SCOPE_GLOBAL_RESERVE_WEIGHT=0.2` |
| `S2H` | + heading lexical scoring | `HEADING_LEXICAL_ENABLED=true`, `HEADING_LEXICAL_WEIGHT=0.20` |
| `S2HR` | + metadata-aware rerank pair enrichment | `RERANK_PAIR_ENRICHMENT_ENABLED=true` |
| `S3` | + retrieval_text reindex + full enrichment | `EVAL_RETRIEVAL_TEXT_MODE=title_context_filename` |

Legacy variants (A0/A1/B1/G0-G3/R1/R2/P1-P3/F1/S0) are retained for backward compatibility.

## Diagnostics (v4 Five-Category Classification)

| Category | Meaning |
| --- | --- |
| `file_recall_miss` | Correct file not in candidates at all |
| `page_miss` | Correct file in top5 but wrong page |
| `ranking_miss` | Correct candidate exists but rerank dropped it out of top5 |
| `hard_negative_confusion` | All top5 from hard negative files |
| `low_confidence` | Top1 score below threshold |

Classification priority is deterministic: `hard_negative_confusion` first, then `file_recall_miss`, `ranking_miss`, `page_miss`, and `low_confidence`. This keeps hard-negative failures visible even when they also imply file recall failure.

## Miss 量化分析

`scripts/analyze_rag_misses.py` consumes `miss_analysis.jsonl` or `results.jsonl` and writes `miss_analysis_report.json` plus `miss_analysis_report.md`.

Required rollups:

- Five-category counts and examples.
- CandidateRecall buckets: `0`, `0<x<0.5`, `>=0.5`, `unknown`.
- Fuzzy match histogram for `《...》` title text versus expected filename.
- Hard-negative/product-family confusion table.
- Anchor hit rate.
- `rerank_drop_top20` sample list.
- False-retrieval document top10.

## Suggested Commands

Smoke check on the current index:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile smoke --variants B0 --skip-reindex --limit 1 --run-id rag-phase0-smoke
```

Phase 2A frozen benchmark on the current index:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile frozen --variants B0,R1,R2 --skip-reindex --run-id rag-phase2a-frozen
```

Phase 2B sparse/RRF benchmark on the selected Phase 2A baseline:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile frozen --variants P1,P2,P3 --skip-reindex --run-id rag-phase2b-sparse
```

Graph fallback/confidence evaluation:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile frozen --variants B0,F1 --skip-reindex --mode graph --run-id rag-phase25-graph
```

Gold confirmation after selecting a frozen winner:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile gold --variants <selected-variant> --skip-reindex --run-id rag-phase2-gold-confirm
```

## Retrieval Variants

- `B0`: current title-context baseline.
- `R1`: `RERANK_TOP_N=20`.
- `R2`: `RAG_CANDIDATE_K=80`, `RERANK_TOP_N=30`, `MILVUS_SEARCH_EF=128`.
- `P1`: current selected Phase 2A baseline, now B0, plus `MILVUS_SPARSE_DROP_RATIO=0.1`.
- `P2`: current selected Phase 2A baseline, now B0, plus `MILVUS_RRF_K=100`.
- `P3`: current selected Phase 2A baseline, now B0, plus `MILVUS_SPARSE_DROP_RATIO=0.1` and `MILVUS_RRF_K=100`.
- `F1`: current selected retrieval baseline, now B0, with confidence gate and fallback enabled for graph evaluation.
- `S0`: structure rerank comparison on top of the retrieval baseline.

Do not select a new default from one metric alone. Prefer variants that improve Page@5, KeywordReq@5, MRR, or ID recall without increasing hard-negative hits or p95 latency materially.
If CUDA is unavailable, treat high-cost local rerank variants such as `R2` as diagnostic for latency and record the run as CPU-only in `config.json`. For CPU smoke or diagnostic runs, set `RERANK_CPU_TOP_N_CAP=10` so variants that request 20/30 rerank candidates cannot accidentally dominate the run.

## Runtime Modes

- `--mode retrieval`: direct retrieval evaluation. This is the default and is used for Phase 1 and Phase 2.
- `--mode graph`: executes the RAG graph to measure confidence fallback/rewrite behavior. Reports include initial vs final retrieval hit rate, fallback trigger/help/hurt rates, rewrite strategy counts, and graph latency.

## Report Files

Each run writes:

- `results.jsonl`: per-sample retrieval rows.
- `summary.json`: aggregate metrics.
- `summary.md`: readable summary.
- `miss_analysis.jsonl`: misses with stage candidates and diagnostic suggestions.
- `config.json`: dataset hash, schema versions, variants, env knobs, git status.

## Phase E: End-to-End Answer Evaluation (S3 后实施)

对齐 all-in-rag 评估三元组：

| Layer | Metric | Method |
| --- | --- | --- |
| Context Relevance | File@5, CandRecall, MRR | Phase 0-D 已覆盖 |
| Faithfulness / Groundedness | 每个 claim 是否可由 top5 上下文证实 | LLM-as-judge (FAST_MODEL), binary grounding per claim |
| Answer Relevance | 最终答案是否直接、完整回答原始问题 | LLM-as-judge, 0-1 score |
| Citation Coverage | 答案中引用的文件/页码是否覆盖 gold 标注 | 自动比对 expected_files/pages |

Phase E 框架设计：
1. 在 `rag_pipeline.py` 的 `generate_answer_node` 后增加评估节点
2. 使用 FAST_MODEL 做 LLM-as-judge，控制延迟增量 < 500ms
3. 评估结果写入 trace 字段 `answer_eval`
4. 扩展 `evaluate_rag_matrix.py` 支持 `--mode answer-eval`
5. S3 达标后启动实施

## Roadmap

以下项目本轮不做，入档备查：

- `gold_chunk_ids / root_ids` 回填：产出 index-aligned qrels，使 Chunk@5/Root@5 恢复可用
- Fallback 以"结构化 query decomposition"重新设计：替代当前 HyDE/step-back 单路重写
- Reranker 用业务语料微调：提升 hard negative 区分能力
- Embedding 升级：bge-m3 -> bge-m3-unsupervised 或更大模型
- LLM planner 替换规则解析器：QueryPlan.intent_type 实现
- Graph RAG / 多跳推理：跨文档推理场景支持
