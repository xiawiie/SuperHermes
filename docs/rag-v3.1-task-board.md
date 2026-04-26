# RAG v3.1 Task Board

Date: 2026-04-26
Status: Complete

## Execution Board

| ID | Task | Status | Evidence |
| --- | --- | --- | --- |
| PR1 | Evaluation split + saved-row regression | Done | `scripts/rag_eval/metrics.py`, `scripts/rag_eval/regression.py`, `--summarize-results-jsonl`; saved-row regression passed |
| PR2 | Chunk/Root qrel contract skeleton | Done | `scripts/rag_qrels.py`; strict/canonical match, conflict policy, coverage gates |
| PR3 | Backend trace/types extraction and connection | Done | `backend/rag_types.py`, `backend/rag_trace.py`; `rag_utils.py` uses shared trace/type helpers |
| PR4 | Behavior-preserving backend split | Done | `backend/rag_retrieval.py`, `backend/rag_rerank.py`, `backend/rag_context.py`, `backend/rag_confidence.py`; compatibility wrappers preserved |
| PR5 | Chunk pool + auto alignment + review dataset | Done | `.jbeval/datasets/rag_chunk_gold_v2*.jsonl/json`; 87/125 aligned |
| PR6 | Chunk/Root qrels in evaluation report | Done | Full qrel eval reports Chunk@5, Root@5, ChunkMRR, RootMRR |
| A | Full code assessment, cleanup, robustness review | Done | `ruff`, `git diff --check`, compileall, full pytest passed |
| B | Design/process/RAG effect review and next plan | Done | `docs/rag-v3.1-project-report-20260426.md` |
| C | Full RAG performance evaluation | Done | `.jbeval/reports/rag-v3.1-full-gold-qrels-refactor-20260426/summary.md` |

## Verification Board

| Check | Status | Evidence |
| --- | --- | --- |
| Unit/integration tests | Done | `uv run pytest tests` -> 220 passed, 1 warning |
| Lint | Done | `uv run ruff check backend scripts tests` -> passed |
| Whitespace validation | Done | `git diff --check` -> passed |
| Python compile | Done | `uv run python -m compileall backend scripts` -> passed |
| Qrel generation | Done | `align_rag_chunk_gold.py` wrote 125 v2 rows |
| Qrel smoke | Done | `.jbeval/reports/rag-boundary-qrels-v2-smoke-20260425-refactor/summary.md` |
| Full qrel matrix | Done | 8 variants x 125 rows = 1000 rows |
| Saved-row regression | Done | `.jbeval/reports/saved-row-regression-full-qrels-20260426/summary_regression.json` |

## Current Remaining Risks

| Risk | Status | Next Action |
| --- | --- | --- |
| Chunk/Root qrels are automatic draft labels, not human approved | Open | Human review 30-50 rows first, then expand |
| `V3F` profile collapsed in the latest full qrel run | Open | Rebuild/verify `v3_fast` index and profile before treating as production data |
| `GS2HR` behavior changed strongly between older no-qrel and latest qrel run | Open | Run isolated golden trace comparison before promoting |
| Main evaluation CLI is still large after partial split | Open | Next refactor should move reporting/preflight/runner with saved-row regression locked |
| Generated `.jbeval/reports` data is large | Open | Commit only selected summaries or define report retention policy |
