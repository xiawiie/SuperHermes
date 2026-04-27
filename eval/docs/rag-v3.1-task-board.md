# RAG v3.1 Closure Task Board

Date: 2026-04-26
Status: In progress

## Release Boundary

v3.1 closes a qrel-governed, traceable, reproducible retrieval-evaluation baseline. Broad backend package restructuring, GraphRAG, HyDE, and online fallback activation are post-tag work unless explicitly marked experimental and fully tested.

## Cleanup Plan

Behavior lock comes first. Each cleanup/refactor pass must run targeted tests plus the saved-row regression when evaluation behavior can be affected.

| Pass | Scope | Smell removed | Verification |
| --- | --- | --- | --- |
| 1 | `scripts/evaluate_rag_matrix.py` config block | Large mixed-responsibility module | saved-row summary regression, targeted tests, ruff, compileall |
| 2 | Stream 2 profile diagnosis | Missing diagnostic boundary for V3F collapse | unit tests, static diagnosis report, live Milvus diagnosis when service is available |
| 3 | Eval module split | Preflight/sample/reporting/io mixed into runner | saved-row regression after each extraction |
| 4 | Qrel review tooling | Draft qrels lack auditable v2.1 review state | fake-judge unit tests, immutable v2 input check |
| 5 | Golden trace/page metrics | Metrics are hard to attribute before/after rerank | unit tests, convergence smoke |
| 6 | Post-tag backend cleanup | Flat backend modules and dead root artifacts | full tests, ruff, compileall, git diff --check |

## Execution Board

| ID | Task | Status | Evidence / Output |
| --- | --- | --- | --- |
| P0.1 | Confirm repo state and behavior lock | Done | `test_evaluate_rag_matrix.py` + `test_rag_eval_regression.py`: 38 passed; saved-row regression baseline has `diff_count: 0` |
| P0.2 | Add Phase 0 boundary: release-critical before broad cleanup | Done | This board separates release closure from post-tag backend restructuring |
| S3.1 | Extract eval variants/config module first | Done | `scripts/rag_eval/variants.py`; main CLI reexports configs; saved-row regression `saved-row-regression-variants-extract-20260426` has `diff_count: 0` |
| S2.1 | Add V3F profile diagnosis script | Done | `scripts/diagnose_variant_profile.py`, `tests/test_diagnose_variant_profile.py`; 42 targeted tests passed |
| S2.2 | Run V3F static diagnosis | Done | `.jbeval/reports/rag-v3.1-v3f-diagnosis-20260426/diagnosis-static.json`; live Milvus query timed out, static report recommends rebuild |
| S1.1 | Implement qrel v2.1 precheck/review/apply CLI | Done | `scripts/review_rag_qrels.py`, `tests/test_review_rag_qrels.py`; fake/noop judge path covered |
| S1.2 | Produce machine review artifacts | Done | `.jbeval/qrel_reviews/precheck*.json*`, `human_review_queue.md`, `llm_review_suggestions.jsonl` |
| S1.3 | Apply human decisions to freeze qrel v2.1 | Todo | `.jbeval/qrels/rag_chunk_gold_v2.1.jsonl`, `v2_to_v2.1_diff.json` |
| S3.2 | Extract sample metrics | Todo | `scripts/rag_eval/sample_metrics.py`, saved-row regression |
| S3.3 | Extract preflight/reporting/io | Todo | `scripts/rag_eval/preflight.py`, `reporting.py`, `io.py`, regression after each |
| S4.1 | Add qrel diff explanation output | Todo | `.jbeval/reports/rag-v3.1-qrel-diff/qrel_diff_explanation.json` |
| S4.2 | Add page-rank before/after rerank metrics | Todo | `results.jsonl` and `summary.md` include page rank delta aggregates |
| S4.3 | Add golden trace analysis | Todo | `scripts/golden_trace_analysis.py`, 36 trace rows |
| S4.4 | Run final convergence matrix | Todo | final 8/7 variant report, 0 errors |
| S5.1 | Run non-blocking page miss/fallback experiments | Todo | only after Stream 4; excluded from tag if unstable |
| S6.1 | Update report/README/.env.example and tag v3.1 | Todo | annotated tag after release commit |
| H1 | Post-tag housekeeping and backend structure cleanup | Todo | archive root docs, move stray tests, update `.gitignore`, architecture doc |

## Verification Board

| Check | Status | Evidence |
| --- | --- | --- |
| Targeted eval tests | Pass | `uv run pytest tests\test_evaluate_rag_matrix.py tests\test_rag_eval_regression.py` -> 38 passed |
| Variants extraction regression | Pass | `.jbeval/reports/saved-row-regression-variants-extract-20260426/summary_regression.json` -> `diff_count: 0` |
| V3F diagnosis tests | Pass | `uv run pytest tests\test_diagnose_variant_profile.py` -> 4 passed |
| Qrel review tests | Pass | `uv run pytest tests\test_review_rag_qrels.py` -> 6 passed |
| Qrel precheck | Open | `pre_check_pass=83`, `pre_check_fail=42`; human review required before v2.1 freeze |
| Focused ruff | Pass | `uv run ruff check scripts\diagnose_variant_profile.py tests\test_diagnose_variant_profile.py scripts\rag_eval\variants.py scripts\evaluate_rag_matrix.py` |
| Focused compileall | Pass | `uv run python -m compileall scripts\diagnose_variant_profile.py scripts\evaluate_rag_matrix.py scripts\rag_eval tests\test_diagnose_variant_profile.py` |
| Full tests | Pass | `uv run pytest tests` -> 230 passed, 1 third-party warning |
| Full ruff | Pass | `uv run ruff check backend scripts tests` -> passed |
| Full compileall | Pass | `uv run python -m compileall backend scripts` -> passed |
| Whitespace | Pass | `git diff --check` -> passed, CRLF warnings only |

## Current Risks

| Risk | Status | Mitigation |
| --- | --- | --- |
| V3F live Milvus diagnosis timed out | Open | Keep V3F out of baseline claims until live diagnosis/rebuild passes hard gates |
| qrel v2.1 still not produced | Open | Next implementation stream is immutable qrel review CLI |
| Eval runner remains large | Open | Continue module extraction one module at a time with saved-row regression |
| Broad backend restructuring conflicts with v3.1 scope | Controlled | Defer directory restructuring to post-tag housekeeping unless needed for release-critical behavior |
