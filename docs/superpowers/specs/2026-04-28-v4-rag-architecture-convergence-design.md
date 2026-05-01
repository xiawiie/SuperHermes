# V4 RAG Architecture Convergence Design

**Date:** 2026-04-28
**Status:** Draft for review
**Scope:** Historical mode-routing design note. Current profile naming is defined only in `docs/rag-profile-naming.md`.

## Problem

The current V4 implementation has the right high-level direction, but the system is not yet trustworthy as an execution architecture. The biggest issue is not one weak algorithm. It is semantic mismatch:

- Some configuration flags imply behavior that is not implemented.
- Trace fields can describe modes, confidence, or plans without proving those values controlled execution.
- Deep Mode can produce user-visible output that looks like an answer even though synthesis and citation verification are not complete.
- Mode-routing evaluation profiles do not currently inherit the `K2 / I2` collection/profile/BM25 state cleanly, so comparisons against `K2 / I2 / M0` can be invalid.
- FAST and Deep are high-risk paths and should not become user-visible active behavior before the surrounding gates are reliable.

The convergence principle is:

> First make the system honest, then make trace explainable, then make plans executable, then evaluate quality.

Mode routing is not a replacement for `K2 / I2 / M0`. It is an outer control layer around the stable `K2` retrieval body: explainable routing, auditable plans, controlled downgrade, and strictly comparable evaluation.

## Goals

1. Make current mode-routing flags, trace fields, and user-visible output honest.
2. Preserve `K2 / I2 / M0` behavior during cleanup and trace propagation phases.
3. Define a single trace schema that distinguishes suggested, planned, applied, executed, downgraded, and disabled behavior.
4. Wire `RetrievalPlan` into existing retrieval/rerank parameter boundaries without creating a parallel pipeline.
5. Repair `M1`/`M2` evaluation profiles so shadow/active runs compare against the same `K2 / I2` corpus, profile, collection, BM25 state, embedding model, and reranker model.
6. Keep FAST and Deep active paths disabled until their gates are proven by tests and gold evaluation.

## Non-Goals

- Do not implement Deep LLM synthesis in this work.
- Do not enable Deep active execution.
- Do not enable FAST active execution by default.
- Do not replace `K2` retrieval or introduce a second pipeline.
- Do not add dependencies.
- Do not add generation-labeled production modules such as `pipeline_v2.py`, `classifier_v4.py`, or `confidence_v2.py`.
- Do not draw quality conclusions from mode-routing eval until the preflight comparability contract passes.

## Design Principles

### No Hidden Behavior

Any user-visible or trace-visible field must be true:

```text
field says X => system actually did X
```

If the system did not execute the behavior, the field must say so with a precise name:

```text
suggested_X
planned_X
plan_applied=false
not_applied_reason
```

### Shadow Isolation

Shadow mode may compute classifier, confidence, and plan diagnostics, but it must not alter retrieval behavior.

Required invariant:

```text
routing_mode=shadow
execution_mode unchanged
retrieval parameters unchanged
plan_applied=false
answer unchanged
```

### FAST Disabled Isolation

Disabling FAST must affect both execution mode and effective plan.

Required invariant:

```text
RAG_FAST_ENABLED=false
=> effective_execution_mode=STANDARD
=> effective_plan=STANDARD_PLAN
```

### Deep Disabled Isolation

Deep suggestion is not Deep execution.

Required invariant:

```text
Deep disabled or suggest-only
=> no synthesized Deep answer
=> no "Deep Mode Answer" block
=> trace may include suggested_mode=DEEP and deep_disabled_reason
```

### Eval Comparability

`M1`/`M2` vs `K2 / I2 / M0` quality comparison is valid only when these match:

```text
collection
profile
BM25 state
corpus
embedding model
reranker model
retrieval text mode
```

If they do not match, the eval must fail before retrieval starts.

## Component Boundaries

### `backend/rag/modes.py`

Responsibility:

- Extract `QueryFeatures`.
- Produce `ModeVerdict`.
- Resolve `RetrievalPlan`.
- Explain routing and downgrade reasons.

It must not call Milvus, rerankers, LLMs, chat tools, or frontend code.

Expected additions:

- `classification_reason`
- `routing_decision_reason`
- `execution_decision_reason`
- `fast_disabled_reason`
- `deep_disabled_reason`
- `plan_source`
- `plan_applied`

### `backend/rag/utils.py`

Responsibility:

- Remain the public retrieval boundary through `retrieve_documents()`.
- Parse `QueryPlan`.
- Invoke mode classification when configured.
- Resolve an effective plan.
- Execute retrieval/rerank/confidence using the effective plan only after the plan-execution phase.
- Return full mode/plan/confidence meta.

In phases 1 and 2, `RetrievalPlan` may be computed but must not alter retrieval parameters. Trace must therefore include `plan_applied=false`.

In phase 3, `RetrievalPlan` may control only:

```text
candidate_k
ce_input_k
ce_top_n
```

No new scoring policy, fallback pipeline, or parallel mode-specific retrieval implementation is allowed.

### `backend/rag/confidence.py`

Responsibility:

- Produce structured confidence verdicts.
- Return answerability, confidence score, risk score, reasons, suggested mode, and clarification state.

It must not format user output, execute Deep, retry retrieval, or mutate fallback graph control by itself.

### `backend/rag/pipeline.py`

Responsibility:

- Keep LangGraph compatibility.
- Keep fallback disabled by default.
- Aggregate retrieval meta into `rag_trace` without changing execution behavior in phase 2.

The graph trace must expose the same mode/plan/confidence facts returned by `retrieve_documents()`.

### `backend/chat/tools.py`

Responsibility:

- Enforce one external `search_knowledge_base` call per turn.
- Format standard retrieved chunks.
- Preserve Deep suggestion as a suggestion only.

It must not output a `Deep Mode Answer` until a later Deep synthesis design implements answer generation and citation verification.

### `backend/chat/deep_mode.py`

Responsibility for this convergence work:

- Keep rule-only decomposition helpers.
- Keep coverage/evidence accumulation helpers.
- Keep no-recursion invariant tests through `mode_override="STANDARD"` and `allow_deep=False`.
- Keep citation verifier as a testable utility, but do not expose it as active production answer gating until synthesis exists.

## Trace Schema

The converged trace should include these fields where applicable:

```text
mode_initial
mode_final
routing_mode
execution_mode
suggested_mode
deep_executed
classification_reason
routing_decision_reason
execution_decision_reason
downgrade_reason
upgrade_reason
fast_disabled_reason
deep_disabled_reason
plan_source
plan_applied
retrieval_plan
effective_retrieval_plan
answerable
confidence_score
risk_score
confidence_reasons
needs_clarification
clarification_question
```

Profile naming is no longer defined in this historical design note. The single
source of truth is `docs/rag-profile-naming.md`; old names in this file are
legacy aliases only.

Trace field semantics for this design:

- `suggested_mode` means classifier or confidence recommends a mode.
- `execution_mode` means what the system actually executed.
- `retrieval_plan` means the resolved policy object.
- `effective_retrieval_plan` means the plan that actually controlled parameters.
- `plan_applied=false` is mandatory until phase 3.
- `deep_executed=false` is mandatory while Deep remains suggest-only.

## Flag Semantics

| Flag | Meaning During Convergence | Behavior |
| --- | --- | --- |
| `RAG_MODE_SHADOW_ENABLED` | Compute classifier and trace only | Must not change execution |
| `RAG_MODE_ROUTING_ENABLED` | Allow routing decisions to choose an effective mode/plan | In phase 2 still no parameter changes; in phase 3 plan may apply |
| `RAG_FAST_ENABLED` | Allow FAST effective plan after routing gates | Defaults off; disabled means STANDARD plan |
| `RAG_DEEP_MODE_ENABLED` | Allow Deep suggestion and future controller entry | During this work, no Deep answer execution |
| `RAG_DEEP_SUGGEST_ONLY` | Keep Deep as recommendation only | Defaults true and must be honored |
| `RAG_DEEP_LLM_PLANNER_ENABLED` | Reserved for later Deep work | No-op/reserved in this scope |
| `RAG_DEEP_CITATION_VERIFIER_ENABLED` | Reserved until Deep synthesis exists | No-op/reserved in this scope |
| `RAG_TRACE_RETENTION_PROFILE` | Trace retention policy | Metadata-only unless retention behavior is implemented |
| `RAG_CONFIDENCE_STRUCTURED_ENABLED` | Add structured confidence meta | Must not change legacy fallback semantics by itself |

Unimplemented or reserved flags must be documented as no-op, telemetry-only, metadata-only, or reserved. They must not imply behavior that the system does not perform.

## Data Flow

```text
Agent
  -> search_knowledge_base(query) once
    -> run_rag_graph(query)
      -> retrieve_documents(query)
        -> QueryPlan
        -> ModeClassifier
        -> RetrievalPlan
        -> retrieval/rerank/confidence
      -> aggregate rag_trace
    -> format retrieved chunks and optional Deep suggestion
```

Deep suggestion flow:

```text
classifier/confidence suggests DEEP
  -> trace suggested_mode=DEEP
  -> deep_executed=false
  -> tool may say Deep mode is suggested but not executed
  -> no Deep answer block
```

## Phases

### Phase 1: Honesty Cleanup

Purpose: clear debt without changing retrieval behavior.

Allowed changes:

- Fix lint, compile, and obvious test failures.
- Remove unused imports.
- Fix undefined variables.
- Stop outputting `Deep Mode Answer` from the tool.
- Mark no-op/reserved flags as such in code comments, config comments, or docs.
- Keep FAST and Deep active behavior off.

Forbidden changes:

- Do not alter retrieval/rerank behavior.
- Do not change `K2 / I2 / M0` eval results.
- Do not wire plan into execution.

Acceptance:

```powershell
uv run ruff check backend/ scripts/ tests/
uv run pytest tests/ -q
uv run python -m compileall backend scripts
```

Deep suggestion must be user-visible only as suggestion/rationale, not as answer.

### Phase 2: Trace Schema And Propagation

Purpose: make behavior explainable without changing execution.

Allowed changes:

- Add and normalize trace fields.
- Propagate `retrieve_documents()` meta into `run_rag_graph()` trace.
- Propagate trace to tool/frontend consumers.
- Add flag matrix tests.
- Add `plan_applied=false`.

Forbidden changes:

- Do not make `RetrievalPlan` alter retrieval/rerank parameters.
- Do not enable FAST active behavior.
- Do not enable Deep active behavior.

Acceptance:

- Shadow mode does not change execution.
- FAST disabled implies `execution_mode=STANDARD` and effective standard plan.
- Deep suggested does not imply Deep executed.
- Existing gold eval behavior remains unchanged.

### Phase 3: RetrievalPlan Execution Wiring

Purpose: make plans actually control bounded parameters.

Allowed changes:

- Let `RetrievalPlan` control `candidate_k`, `ce_input_k`, and `ce_top_n`.
- Pass plan parameters through existing retrieval/rerank helpers.
- Add tests proving plan parameters reach execution.
- Add STANDARD equivalence tests.
- Add FAST disabled contamination tests.
- Add conservative FAST downgrade/retry tests.

Forbidden changes:

- Do not introduce new pipelines.
- Do not introduce new rerank scoring policy.
- Do not change the `K2 / I2 / M0` standard baseline outside the explicit plan boundary.
- Do not enable FAST by default.

FAST behavior must remain conservative:

```text
FAST suggested + FAST disabled => execute STANDARD
FAST suggested + low confidence => execute STANDARD or retry STANDARD
FAST executed + high risk => retry/downgrade STANDARD
```

Trace examples:

```text
suggested_mode=FAST
execution_mode=STANDARD
downgrade_reason=fast_disabled
plan_applied=true
effective_retrieval_plan=STANDARD_PLAN
```

or:

```text
execution_mode_initial=FAST
execution_mode_final=STANDARD
retry_reason=low_confidence
```

Acceptance:

- STANDARD plan remains equivalent to current `K2 / I2 / M0` behavior.
- FAST disabled cannot apply compact parameters.
- Plan parameters are observable in trace and tested at execution boundaries.

### Phase 4: Mode Eval Comparability Repair

Purpose: make `M1`/`M2` comparisons against `K2 / I2 / M0` valid before drawing
conclusions.

Allowed changes:

- Repair mode-routing eval profiles so they inherit the `K2 / I2` collection,
  profile, BM25 state, and retrieval text mode.
- Add preflight comparability checks.
- Record routing state with `M0`, `M1`, or `M2` plus explicit flags instead of
  inventing new variant names in this document.

Forbidden changes:

- Do not run invalid comparisons with warning-only failures.
- Do not enable active/FAST from an eval report that failed comparability preflight.

Acceptance:

```powershell
uv run python scripts/evaluate_rag_matrix.py --dataset eval/datasets/rag_doc_gold.jsonl --mode retrieval --skip-reindex --run-id mode-comparability-check
```

The eval must either:

- pass preflight and produce comparable `M1`/`M2` vs `K2 / I2 / M0` metrics, or
- fail before retrieval with a clear comparability error.

## Test Plan

### Unit Tests

- Mode scoring and hard blocks.
- Plan resolution with shadow/routing/FAST flags.
- Confidence verdict fields.
- Reserved/no-op flag behavior where applicable.
- Deep disabled and suggest-only invariants.

### Boundary Tests

- `retrieve_documents()` returns complete mode/plan/confidence meta.
- Shadow mode leaves execution and effective plan unchanged.
- FAST disabled returns STANDARD effective plan.
- Phase 3 plan parameters reach candidate/rerank execution.

### Pipeline And Tool Tests

- `run_rag_graph()` propagates mode/plan/confidence trace fields.
- Tool output includes Deep suggestion only, never Deep answer, while Deep synthesis is absent.
- One-call tool guard remains effective.

### Eval Tests

- `evaluate_mode_classifier.py` remains green.
- Mode-routing profiles pass comparability preflight before running gold retrieval eval.
- Invalid collection/profile/BM25 combinations fail hard.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Trace looks correct but behavior is not | Include `plan_applied` and execution/effective plan fields |
| FAST contaminates STANDARD while disabled | Assert effective plan, not only mode string |
| Shadow mode changes quality | Require zero behavior changes in phases 1 and 2 |
| Deep suggestion still looks like an answer | Remove `Deep Mode Answer` formatting until synthesis exists |
| Mode-routing eval compares different corpora/states | Hard-fail preflight if `M1`/`M2` and `K2 / I2 / M0` infrastructure differs |
| Parallel pipeline sprawl | Parameterize existing retrieval/rerank helpers only |
| EXP_C5-style candidate compression regression | Keep FAST off by default and gate compact plans on gold eval |

## Completion Criteria

The convergence work is complete when:

- Lint, tests, and compileall pass.
- Current misleading Deep answer output is gone.
- Trace differentiates suggested/planned/applied/executed/downgraded behavior.
- `RetrievalPlan` can be computed without hidden behavior in phase 2 and applied explicitly in phase 3.
- FAST disabled guarantees STANDARD effective execution and plan.
- Deep disabled/suggest-only guarantees no synthesized answer.
- Mode-routing eval profiles are comparable to `K2 / I2 / M0` or fail before retrieval.
- No active routing or FAST default is enabled without gold-gate evidence.

