# Pipeline-First Agentic RAG Architecture

**Date:** 2026-04-27  
**Status:** Design draft  
**Target:** Replace the serial LLM-gated retrieval path with a mode-aware pipeline while keeping the codebase small, reversible, and aligned with the current SuperHermes architecture.

---

## Executive Summary

SuperHermes already has strong RAG building blocks: QueryPlan, scoped hybrid retrieval, layered rerank, confidence gates, diagnostics, qrel-aware evaluation, and streaming RAG observability. The next step is not a large rewrite. The next step is to make the existing pipeline mode-aware and move expensive agentic behavior out of the common path.

The desired runtime behavior:

- Simple precise questions use a FAST retrieval plan.
- Normal document questions use a STANDARD retrieval plan.
- Complex multi-hop, comparison, exhaustive, or temporal questions use an internal DEEP controller.
- FAST and STANDARD retrieval do not call an LLM.
- DEEP may call an LLM for planning and synthesis, but all sub-retrieval calls run as STANDARD and cannot recursively trigger DEEP.

The core principle stays:

> The pipeline handles deterministic retrieval by default. Agentic reasoning is reserved for complex queries and is budgeted, observable, and non-recursive.

This design intentionally avoids versioned production module names such as `pipeline_v2` or `confidence_v2`. New files are allowed only when they create a clean boundary that would otherwise bloat an existing module.

---

## Current Evidence

### Proven Strengths

- `V3Q + FP16` is the strongest low-risk performance improvement already measured: quality is preserved while P50 drops to roughly 1000 ms on the 125-row gold run.
- QueryPlan and document scope matching already provide deterministic routing signals.
- Layered rerank already provides useful trace fields and candidate-stage observability.
- The evaluation harness already supports File/Page/Chunk/Root metrics, qrels, saved-row regression, and fallback analysis.

### Known Failure To Avoid

The 125-row gold run showed that aggressive L1/CE compression can hurt quality:

- `EXP_C5` dropped File@5 from `0.992` to `0.864`.
- Correct files were still present before rerank, but CrossEncoder ranking dropped them from top-5.
- Therefore FAST/STANDARD plans must not blindly shrink CE input for broad or ambiguous queries.

Design consequence:

- STANDARD starts quality-first, using the proven V3Q FP16 profile.
- FAST is enabled only for high-confidence single-fact queries.
- Any candidate compression must pass the 125-row gold gates before it becomes default.

---

## Goals

1. Reduce average retrieval latency without lowering retrieval quality.
2. Remove LLM Grader and LLM Router from FAST/STANDARD hot paths.
3. Add DEEP mode for multi-hop and synthesis-heavy questions.
4. Preserve current observability and improve it with mode, plan, and confidence traces.
5. Keep production code clean: minimal files, clear names, no version suffixes.
6. Keep every phase reversible behind configuration flags.

## Non-Goals

- Do not rewrite the whole RAG stack.
- Do not introduce new dependencies.
- Do not make the Agent call `search_knowledge_base` multiple times.
- Do not remove LangGraph from the main path until shadow comparison proves parity.
- Do not make DEEP the default answer path.

---

## Naming And File Discipline

Production modules should describe responsibilities, not generations.

Avoid:

- `pipeline_v2.py`
- `classifier_v4.py`
- `confidence_v2.py`
- `retrieval_plan_v4.py`

Prefer:

- `modes.py`
- `confidence.py`
- `pipeline.py`
- `tools.py`
- `deep_mode.py`

Minimal new files:

| File | Reason |
| --- | --- |
| `backend/rag/modes.py` | A cohesive boundary for query features, mode classification, retrieval plans, and shadow metrics. Keeping this in `query_plan.py` would mix document scope parsing with mode policy. |
| `backend/chat/deep_mode.py` | A separate chat-layer orchestration boundary for DEEP planning, internal sub-retrieval, evidence accumulation, and synthesis. Keeping this in `agent.py` or `tools.py` would bloat already busy files. |

No other new production files are planned at the start. If a module grows beyond a clear single purpose, extract later with evidence.

---

## Runtime Architecture

```text
User message
  -> Agent retrieval decision
  -> search_knowledge_base tool or attached-context retrieval
  -> Mode classifier
  -> Retrieval plan
  -> Existing retrieve_documents path
       QueryPlan
       Dense + sparse retrieval
       Optional scoped/global blend
       Layered rerank when enabled
       Structure rerank
       Confidence verdict
  -> If STANDARD/FAST answerable: return evidence to Agent
  -> If DEEP required: internal deep controller
       Plan subqueries
       Run STANDARD sub-retrievals
       Accumulate evidence
       Synthesize answer with citations
```

The Agent still sees one tool call. DEEP is an internal orchestration path, not multiple tool calls from the Agent.

---

## Modes

| Mode | Purpose | Hot-path LLM | Retrieval calls | Initial target |
| --- | --- | ---: | ---: | --- |
| FAST | Precise single-fact lookup with strong scope or anchor | 0 | 1 | P50 under 600 ms after calibration |
| STANDARD | Default document retrieval path | 0 | 1 | Preserve V3Q FP16 quality, P50 around 1000-1300 ms |
| DEEP | Multi-hop, comparison, exhaustive, temporal, or low-confidence synthesis | 1-2 | 2-3 internal calls | P95 under 6000 ms, traffic under 15% |

Routing rules:

- STANDARD is the safe default.
- FAST requires high confidence before retrieval and high confidence after retrieval.
- DEEP intent overrides FAST.
- User "quick" requests may downgrade DEEP to STANDARD, but never force FAST for complex questions.
- Subqueries inside DEEP always run as STANDARD with `allow_deep=False`.

---

## Mode Classifier

The classifier is rule-based and deterministic. It should build on existing QueryPlan signals instead of creating a second document resolver.

### QueryFeatures

`backend/rag/modes.py` owns a compact feature object:

```python
@dataclass(frozen=True)
class QueryFeatures:
    entities: tuple[str, ...]
    entity_count: int
    has_precise_scope: bool
    has_single_file_match: bool
    has_anchor: bool
    scope_is_global: bool
    has_compare_intent: bool
    has_summary_intent: bool
    has_exhaustive_intent: bool
    has_causal_intent: bool
    has_negation: bool
    has_temporal_evolution_intent: bool
    has_multi_field_extraction: bool
    is_context_reference: bool
    user_explicit_fast: bool
    user_explicit_deep: bool
    question_type: str
```

This intentionally starts smaller than the original draft. Add fields only when tests or trace analysis prove they are needed.

### ModeVerdict

```python
@dataclass(frozen=True)
class ModeVerdict:
    mode: Literal["FAST", "STANDARD", "DEEP"]
    reason: str
    fast_score: float
    deep_score: float
    features: QueryFeatures
```

The first implementation should shadow-run:

- Compute `ModeVerdict`.
- Write it into `rag_trace`.
- Do not alter behavior until classifier evaluation is available.

### Classifier Acceptance

A classifier eval set is required before active routing:

- At least 100 labeled queries.
- Must include precise facts, single-file summaries, comparisons, exhaustive/global questions, temporal-evolution questions, and context-reference questions.
- FAST false-positive rate on complex queries must be below 2%.
- DEEP trigger rate in normal traffic should stay below 15%.

---

## Retrieval Plans

Plans are configuration objects, not new pipelines.

```python
@dataclass(frozen=True)
class RetrievalPlan:
    mode: Literal["FAST", "STANDARD", "DEEP"]
    candidate_k: int
    ce_input_k: int
    ce_top_n: int
    use_layered_rerank: bool
    allow_llm: bool
    allow_deep: bool
    max_wall_time_ms: int
    confidence_profile: str
```

Initial plan mapping:

| Plan | Behavior |
| --- | --- |
| FAST | Uses existing QueryPlan and layered rerank with conservative activation. It may use smaller CE input only for single-file or anchored facts. |
| STANDARD | Uses the proven quality-first profile first: V3Q + FP16, current QueryPlan, scoped hybrid, rerank, structure rerank, confidence gate. |
| DEEP | Not a direct retrieval plan for subqueries. The DEEP controller runs multiple STANDARD sub-retrievals. |

Important:

- Do not promote aggressive EXP_C5-style compression to STANDARD by default.
- Candidate compression belongs behind a plan flag and must pass full gold evaluation.
- Plan values should come from environment/config and trace output, not hardcoded scattered constants.

---

## Retrieval Pipeline Changes

The public retrieval function remains the main boundary:

```python
def retrieve_documents(
    query: str,
    top_k: int = 5,
    context_files: list[str] | None = None,
    *,
    mode_override: str | None = None,
    allow_deep: bool = True,
) -> dict:
    ...
```

Behavior:

1. Parse QueryPlan as today.
2. Classify mode using QueryPlan plus query features.
3. Resolve the RetrievalPlan.
4. Execute existing retrieval/rerank/context/confidence functions with plan-aware parameters.
5. Return the existing shape: `{"docs": ..., "meta": ...}`.
6. Add plan fields to `meta`, including:
   - `mode_initial`
   - `mode_final`
   - `mode_reason`
   - `retrieval_plan`
   - `answerable`
   - `confidence_score`
   - `confidence_reasons`
   - `suggested_mode`
   - `needs_clarification`

This keeps API compatibility while making the result richer.

---

## Confidence Verdict

The current confidence gate should evolve in place inside `backend/rag/confidence.py`.

```python
@dataclass(frozen=True)
class ConfidenceVerdict:
    answerable: bool
    confidence_score: float
    risk_score: float
    reasons: tuple[str, ...]
    suggested_mode: Literal["FAST", "STANDARD", "DEEP"] | None
    needs_clarification: bool
    clarification_question: str | None
```

Upgrade semantics:

- FAST low confidence upgrades to STANDARD immediately.
- STANDARD high risk returns `suggested_mode="DEEP"` to the chat layer.
- Retrieval code does not silently run DEEP during a normal retrieval call.
- If evidence is insufficient and DEEP is not allowed, return an answerability failure with a clarification question or a no-evidence result.

This closes the ambiguity in the earlier design where `mode_final` could become DEEP without specifying whether DEEP actually ran.

---

## Deep Mode

DEEP mode lives in `backend/chat/deep_mode.py`.

It is an internal controller:

1. Accept the original query, context files, conversation context, and classifier verdict.
2. Build up to 3 subqueries.
3. Run each subquery through `retrieve_documents(..., mode_override="STANDARD", allow_deep=False)`.
4. Accumulate evidence with source attribution.
5. Evaluate coverage.
6. Stop early when coverage is sufficient.
7. Synthesize a cited answer with one LLM call.

### Budget

| Budget | Default |
| --- | ---: |
| Max internal retrieval calls | 3 |
| Max planner LLM calls | 1 |
| Max synthesis LLM calls | 1 |
| Max wall time | 6000 ms |
| Max evidence items sent to synthesis | 15 |
| Max characters per evidence item | 600 |

### Subquery Rules

Rule decomposer handles common patterns first:

- Compare A and B -> one subquery per entity/aspect.
- Temporal evolution -> earliest, latest, change.
- Exhaustive global query -> one broad STANDARD query plus optional targeted follow-up if coverage is low.
- Ambiguous complex query -> LLM planner fallback, capped at 3 subqueries.

### Evidence Accumulation

Evidence can start inside `deep_mode.py`. Do not add a separate `evidence.py` until the code becomes reusable outside DEEP.

Evidence item fields:

- `dedupe_key`
- `doc`
- `subquery_ids`
- `target_entities`
- `target_aspects`
- `route_sources`
- `best_score`

### Agent Contract

The existing Agent may still call `search_knowledge_base` at most once per turn.

The tool may internally choose:

- STANDARD retrieval result formatting.
- DEEP controller result formatting.

The Agent is not allowed to make repeated knowledge-base tool calls. This preserves the current guard and avoids tool-call loops.

---

## Tool Output

`backend/chat/tools.py` should cap retrieval output regardless of mode:

| Setting | Default |
| --- | ---: |
| Max docs returned to Agent | 5 |
| Max chars per doc | 800 |
| Include diagnostics | Yes |
| Include trace context for frontend | Yes |

Tool output should include:

- Retrieved chunks.
- Confidence.
- Answerability.
- Covered and missing entities where available.
- Suggested mode when retrieval is not enough.
- Clarification question when needed.

---

## Observability

Add trace fields without breaking current reports.

### Mode And Plan Metrics

- `mode_initial`
- `mode_final`
- `mode_reason`
- `fast_score`
- `deep_score`
- `retrieval_plan`
- `mode_shadow_mismatch`
- `mode_override_used`

### Confidence Metrics

- `confidence_score`
- `risk_score`
- `confidence_reasons`
- `answerable`
- `suggested_mode`
- `needs_clarification`

### Deep Metrics

- `deep_trigger_reason`
- `deep_tool_calls_used`
- `deep_llm_calls_used`
- `deep_coverage_score`
- `deep_missing_entities`
- `deep_missing_aspects`
- `deep_early_stop`
- `deep_timeout`
- `deep_fallback_used`
- `deep_duplicate_evidence_rate`

### Quality Metrics

Keep current evaluation metrics:

- File@5
- File+Page@5
- Chunk@5
- Root@5
- MRR
- NDCG@5
- MAP@5
- hard-negative hit rate
- candidate recall before rerank
- rerank drop rate
- structure drop rate

---

## Implementation Phases

### Phase 0: Low-Risk Baseline

Purpose: take the proven FP16 gain before architecture changes.

Changes:

- Set and document `RERANK_TORCH_DTYPE=float16` for the quality profile.
- Keep behavior otherwise unchanged.

Acceptance:

- Existing tests pass.
- Smoke RAG eval passes.
- No answer-shape change.

### Phase 1: Shadow Mode Classifier

Purpose: collect evidence before routing.

Changes:

- Add `backend/rag/modes.py`.
- Add classifier unit tests.
- Write `ModeVerdict` into trace only.

Acceptance:

- FAST false positives on labeled complex queries below 2%.
- DEEP trigger rate is measurable.
- Existing retrieval results unchanged.

### Phase 2: Plan-Aware Retrieval

Purpose: allow plan parameters without changing public API.

Changes:

- Extend `retrieve_documents` with optional mode/plan controls.
- Add plan fields to meta.
- STANDARD remains quality-first by default.
- FAST is disabled or shadow-only until classifier confidence is proven.

Acceptance:

- Existing tests pass.
- Eval rows remain comparable.
- Trace shows selected plan.

### Phase 3: Structured Confidence

Purpose: replace ambiguous fallback flags with explicit verdicts.

Changes:

- Extend `backend/rag/confidence.py`.
- Convert confidence results into `answerable`, `risk_score`, `suggested_mode`, and `needs_clarification`.
- Keep legacy meta fields until callers are migrated.

Acceptance:

- Existing fallback tests pass.
- Saved-row regression can summarize both legacy and new fields.

### Phase 4: FAST Activation

Purpose: route only safe precise queries to FAST.

Changes:

- Enable FAST for high-confidence single-file or anchored facts.
- FAST low confidence immediately retries STANDARD.

Acceptance:

- 125-row gold quality does not regress beyond gates.
- FAST traffic P50 improves meaningfully.
- FAST-to-STANDARD upgrade rate is tracked.

### Phase 5: Deep Controller

Purpose: support complex multi-hop questions without putting Agent on the main chain.

Changes:

- Add `backend/chat/deep_mode.py`.
- Implement rule decomposer, budget tracker, evidence accumulator, and synthesis.
- Keep subqueries STANDARD and non-recursive.
- Tool still counts as one Agent tool call.

Acceptance:

- Deep eval set passes.
- Deep trigger rate remains below 15% in representative traffic.
- No recursive DEEP calls.
- Budget exhaustion returns partial evidence safely.

### Phase 6: Main-Path Simplification

Purpose: remove serial LLM gating from the common path.

Changes:

- Disable LLM Grader/Router for FAST/STANDARD when confidence verdict is active.
- Keep LangGraph fallback switch while comparing outputs.
- Remove LangGraph from the main path only after shadow parity is proven.

Acceptance:

- Full test suite passes.
- Gold eval passes.
- Graph and non-graph trace comparison is explained.
- Rollback flag remains available for one release.

---

## Acceptance Gates

Quality gates use the 125-row gold dataset when available.

| Metric | Gate |
| --- | --- |
| STANDARD File@5 | no more than 0.3pp below V3Q FP16 baseline |
| STANDARD File+Page@5 | no more than 0.5pp below V3Q FP16 baseline |
| STANDARD Chunk@5 | no more than 0.5pp below baseline on rows with qrels |
| STANDARD Root@5 | no more than 0.5pp below baseline on rows with qrels |
| FAST complex false-positive rate | under 2% |
| DEEP normal-traffic trigger rate | under 15% |
| Error rate | 0.0 on eval runs |
| Recursive DEEP calls | 0 |

Latency gates:

| Mode | Gate |
| --- | --- |
| FAST | P50 under 600 ms after active routing |
| STANDARD | P50 around 1000-1300 ms, P95 under 1800 ms |
| DEEP | P95 under 6000 ms |

If a faster plan misses quality gates, it remains an experiment and does not become default.

---

## File Change Plan

### New Files

| File | Responsibility |
| --- | --- |
| `backend/rag/modes.py` | Query feature extraction, mode scoring, mode verdicts, retrieval plans. |
| `backend/chat/deep_mode.py` | Internal DEEP controller, subquery planning, evidence accumulation, synthesis. |

### Modified Files

| File | Changes |
| --- | --- |
| `backend/rag/utils.py` | Accept optional mode controls, apply retrieval plans, add mode/plan/confidence meta. |
| `backend/rag/confidence.py` | Add structured confidence verdict while preserving legacy meta fields. |
| `backend/rag/layered_rerank.py` | Read plan parameters for L1/L2/L3 quotas where needed. |
| `backend/rag/pipeline.py` | Keep as compatibility path while removing LLM grading/routing from FAST/STANDARD after confidence is active. |
| `backend/chat/tools.py` | Keep one-call guard, cap output, dispatch internally to STANDARD or DEEP. |
| `backend/chat/agent.py` | Update prompt and streaming trace handling for mode-aware retrieval. |
| `backend/config.py` | Add configuration flags and defaults. |
| `scripts/rag_eval/variants.py` | Add shadow/active mode experiment variants without changing existing variants. |
| `tests/*` | Add classifier, plan, confidence, deep controller, and regression tests. |

No broad file split is planned. Extraction is allowed only after a module has a proven second responsibility.

---

## Test Plan

Unit tests:

- Mode classifier examples.
- FAST hard blocks.
- User explicit fast/deep handling.
- Context reference behavior.
- Confidence verdict risk scoring.
- FAST low-confidence retry to STANDARD.
- DEEP subqueries use STANDARD with `allow_deep=False`.
- Tool guard still prevents repeated Agent tool calls.
- Tool output caps docs and characters.

Integration tests:

- `search_knowledge_base` returns legacy-compatible text plus diagnostics.
- Streaming still emits RAG steps.
- Attached context files still constrain retrieval.
- Graph fallback path remains available.

Evaluation:

```powershell
uv run pytest tests/ -q
uv run ruff check backend/ scripts/ tests/
uv run python -m compileall backend scripts
uv run python scripts/evaluate_rag_matrix.py --dataset eval/datasets/rag_doc_gold.jsonl --variants V3Q --mode retrieval --skip-reindex --run-id baseline-check
```

Add focused runs for active mode changes:

```powershell
uv run python scripts/evaluate_rag_matrix.py --dataset eval/datasets/rag_doc_gold.jsonl --variants <mode-shadow>,<mode-active> --mode retrieval --skip-reindex --run-id mode-routing-check
```

---

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| FAST routes complex query | Low recall or wrong answer | Strict FAST blocks, shadow eval, immediate STANDARD retry on low confidence. |
| STANDARD over-compresses candidates | Regression like EXP_C5 | Keep V3Q FP16 quality-first baseline; require full gold gates for compression. |
| DEEP over-triggers | Latency and cost spike | Trigger cap, metrics, explicit thresholds, traffic sampling. |
| DEEP recurses | Budget explosion | `allow_deep=False` on all subqueries, test this invariant. |
| Agent loops tools | Slow or unstable answers | Preserve one-call guard; DEEP runs inside the tool/controller. |
| Confidence miscalibration | Bad answerability decisions | Saved-row regression plus confidence calibration set. |
| More files than needed | Codebase drift | Start with only `modes.py` and `deep_mode.py`; extract later only with evidence. |
| LangGraph removal breaks streaming trace | UX regression | Keep compatibility switch and shadow compare before removal. |

---

## Open Decisions

1. FAST should remain shadow-only until the classifier eval set is created.
2. STANDARD should start from the proven V3Q FP16 behavior, not the aggressive layered compression profile.
3. DEEP synthesis prompt should be citation-first and explicitly state missing evidence.
4. Any future module extraction must remove complexity from an existing file, not create a parallel versioned stack.

---

## Final Design Position

This design keeps the original pipeline-first agentic RAG goal, but changes the execution posture:

- Reuse current assets.
- Add only essential boundaries.
- Avoid versioned module names.
- Prove routing before enabling it.
- Keep Agent behavior bounded.
- Let evaluation gates, not architectural enthusiasm, decide when a faster plan becomes default.
