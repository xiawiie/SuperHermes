# 2026-04-26 Refactor Baseline Anchor

## Purpose

This document freezes the pre-refactor baseline for the full-repo slimming and engineering reorganization work. It is the rollback anchor for phase-level verification.

## Repository Snapshot

- Timestamp: 2026-04-26
- Workspace: `C:/Users/goahe/Desktop/Project/SuperHermes`
- Git status: dirty working tree with existing report/log/cache artifacts and `.omx` state updates (left untouched).

## Baseline Verification Results

### Quick test gate

1. `uv run pytest tests/test_evaluate_rag_matrix.py tests/test_rag_eval_regression.py -q`
   - Result: `38 passed in 0.83s`
2. `uv run pytest tests/test_rag_utils.py tests/test_rag_pipeline_fast_path.py -q`
   - Result: `22 passed, 1 warning in 8.90s`
   - Warning: third-party `jieba` deprecation warning (`pkg_resources`), pre-existing and non-blocking.

### Quality/syntax gate

1. `uv run ruff check backend scripts tests`
   - Result: `All checks passed!`
2. `uv run python -m compileall backend scripts`
   - Result: success (`backend` and `scripts` compiled without syntax failures).

### Smoke evaluation gate

- Command:
  - `uv run python scripts/evaluate_rag_matrix.py --dataset-profile smoke --variants B0 --skip-reindex --limit 1 --run-id baseline-anchor-20260426`
- Status:
  - Command entered a long-running state with no emitted output in the current environment and was terminated to unblock staged work.
- Policy for this refactor stream:
  - Treat smoke eval as an environment-gated check and run it again at each milestone boundary where runtime services are confirmed healthy.

## Rollback Rule for This Stream

If any phase introduces test failures, lint failures, syntax failures, or measurable behavior regressions, roll back to the phase start state and re-apply changes in smaller increments.
