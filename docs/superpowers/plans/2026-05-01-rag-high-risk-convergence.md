# RAG High-Risk Convergence Plan

## Current Direction

This plan tracks four high-risk RAG convergence areas:

1. Candidate strategy and shared rerank boundary.
2. Fallback latency from repeated full retrieval.
3. Agent and direct chat path execution consistency.
4. Unified trace and error semantics.

The candidate strategy boundary is now implemented as the L0-L3 runtime model:

```text
L0 Retrieval
L1 Candidate Selection
L2 Shared Rerank
L3 Shared Postprocess
```

`standard` and `layered` are L0/L1 candidate strategies. They do not own L2/L3. Complete retrieval still exits through `finish_retrieval_pipeline()`.

## Implemented In This Branch

- `backend/rag/runtime_config.py` reads one candidate strategy flag: `RAG_CANDIDATE_STRATEGY`.
- `backend/rag/candidate_strategy.py` defines the shared strategy and rerank execution trace contract.
- `backend/rag/layered_candidates.py` owns layered candidate preset and L1 helper functions only.
- `backend/rag/utils.py` dispatches between standard and layered candidate pools, then uses shared rerank/postprocess.
- `retrieve_candidate_pool()` marks `rerank_execution_mode=skipped_candidate_only`.
- Eval summaries report requested/effective candidate strategy and rerank execution mode.

## Remaining Work

Fallback latency:

- Keep candidate-only fallback behind its existing flag.
- Re-run smoke eval for standard vs layered after this migration.
- Watch `final_rerank_input_count`, latency, hit rate, and fallback helped/hurt rate.

Agent/direct path consistency:

- Keep `RagExecutionPolicy` as the boundary.
- Do not force every path through the same tool call.
- Make trace, citation verification, Deep shadow, and error reporting attach at one shared output boundary.

Error semantics:

- Continue using `StageError` shape from `backend/rag/types.py`.
- Prefer adding structured error code fields over silent fallbacks.
- Keep rerank failures visible in trace while returning ranked candidates when possible.

Deep Mode:

- Keep Deep as an orchestrator over standard RAG, not a separate retrieval system.
- Active Deep answers must remain gated by evidence coverage and citation verification.

## Acceptance Checks

Target tests:

```powershell
python -m pytest -q tests\test_rag_utils.py tests\test_rag_trace.py tests\test_layered_candidates.py tests\test_evaluate_rag_matrix.py
```

Full verification:

```powershell
python -m pytest -q tests
python -m ruff check backend scripts tests
python -m compileall -q backend scripts tests
```

Naming check: grep for obsolete formal names across `backend`, `tests`, `scripts`, and `docs/superpowers/plans`.

Expected: no active runtime, test, script, or current plan references to obsolete formal names.
