# SuperHermes Pipeline-First Agentic RAG Architecture

**Date:** 2026-04-27
**Status:** Draft
**Target:** Replace the serial LLM-gated RAG path with a mode-aware, pipeline-first architecture; add Deep Mode for complex multi-hop queries without turning the main chain into an agent loop.

---

## Executive Summary

Current V3Q behavior is strong but expensive: retrieval can pass through a serial chain of retrieval, LLM grading, LLM routing, query rewrite, and second retrieval. The measured quality ceiling is good, especially with `V3Q + FP16`, but the architecture still mixes deterministic retrieval with agentic fallback logic.

The target redesign keeps the original intent of the draft:

- A rule-based Mode Classifier routes queries to FAST / STANDARD / DEEP.
- FAST and STANDARD retrieval hot paths make zero LLM calls.
- Retrieval stays pipeline-first and reuses current QueryPlan, hybrid retrieval, layered rerank, confidence, diagnostics, and evaluation assets.
- DEEP mode handles complex multi-hop, compare, exhaustive, or temporal-evolution questions through an internal Planner-Controller.
- The Agent still calls the knowledge-base tool at most once per turn; Deep Mode runs inside the tool/controller, not through repeated Agent tool calls.

Core principle:

> Pipeline handles deterministic retrieval. Agentic behavior is reserved for the complex minority path, with hard budgets and traceable evidence.

This revision keeps the original document's architecture, but changes the implementation posture: do not create a parallel versioned stack, do not generate many new files, and do not promote aggressive candidate compression until the full gold evaluation proves it safe.

---

## Existing Evidence And Constraints

### Current Assets To Reuse

SuperHermes already has these production-quality building blocks:

- `backend/rag/query_plan.py`: query parsing, document-scope matching, route hints.
- `backend/rag/utils.py`: retrieval orchestration and public `retrieve_documents` boundary.
- `backend/rag/layered_rerank.py`: split retrieval, L1 candidate selection, adaptive CE sizing, weak structure helpers.
- `backend/rag/confidence.py`: retrieval confidence gate and anchor checks.
- `backend/rag/diagnostics.py`: failure classification.
- `backend/rag/pipeline.py`: current LangGraph compatibility path.
- `scripts/evaluate_rag_matrix.py` and `scripts/rag_eval/*`: qrel-aware evaluation and regression reporting.

The new design should use these components first. A new file is justified only when it prevents an existing file from taking on a second, unrelated responsibility.

### Important Experimental Result

The layered-rerank experiments produced a warning that must shape the design:

- `V3Q + FP16` is a low-risk improvement and can be used as the quality-first baseline.
- `EXP_C5` looked strong on the 44-row frozen set, but on the 125-row gold set dropped File@5 from `0.992` to `0.864`.
- The failure was not candidate recall. Correct files were still present before rerank; CrossEncoder ranking dropped them from top-5.

Design consequence:

- STANDARD must start from the proven V3Q FP16 behavior.
- FAST may compress candidates only for clearly safe single-fact queries.
- Any aggressive L1/CE compression remains experimental until it passes the 125-row gold gates.

---

## Architecture Overview

```text
+----------------------------------------------------------+
| Agent Layer                                               |
| - Decides whether retrieval is needed                     |
| - Calls search_knowledge_base at most once per turn        |
| - Receives either STANDARD/FAST evidence or DEEP result    |
+----------------------------------------------------------+
| Mode Classifier                                           |
| - Rule-based, zero LLM                                    |
| - Uses QueryPlan + query features                         |
| - Starts in shadow mode, then controls routing             |
+----------------------------------------------------------+
| RAG Pipeline                                              |
| FAST:     precise single-fact retrieval, zero LLM          |
| STANDARD: quality-first retrieval, zero LLM                |
| DEEP:     internal planner -> STANDARD sub-retrievals      |
+----------------------------------------------------------+
| Infrastructure                                            |
| - Milvus dense + sparse hybrid retrieval                   |
| - CrossEncoder rerank, FP16 where supported                |
| - Redis / PostgreSQL                                      |
+----------------------------------------------------------+
```

### Key Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| Where does intelligence live? | Pipeline by default, Deep controller for complex queries | Deterministic retrieval should stay stable and testable |
| Serial or parallel retrieval? | Mode-aware pipeline with existing split/scoped retrieval reused | Avoid rebuilding already validated retrieval layers |
| LLM in FAST/STANDARD hot path? | No | Cost, latency, and determinism |
| LLM Grader/Router | Replaced by confidence verdict for FAST/STANDARD | Avoid serial LLM gates on common path |
| Multi-hop strategy | Planner-Controller, not unconstrained ReAct | Bounded execution and easier verification |
| LangGraph | Keep as compatibility/shadow path until parity is proven | Avoid breaking streaming and current callers |
| File structure | Minimal new files, no version-suffixed production modules | Keep codebase clean and avoid parallel stacks |

### Red Lines

1. FAST/STANDARD retrieval stage makes zero LLM calls.
2. Deep subqueries must run as STANDARD with `allow_deep=False`.
3. Deep Mode target traffic stays below 15%.
4. Budgets for tool calls, LLM calls, and wall time are enforced in code.
5. The Agent must not repeatedly call `search_knowledge_base` in one turn.
6. Production module names should not carry generation labels like `pipeline_v2`, `confidence_v2`, or `classifier_v4`.
7. Candidate compression cannot become default unless full gold evaluation passes.

### Configuration And Rollout Controls

All routing changes are controlled by explicit flags. Defaults preserve current behavior.

| Flag | Default | Purpose |
| --- | --- | --- |
| `RAG_MODE_SHADOW_ENABLED` | `false` | Compute classifier verdicts and trace them without changing retrieval. |
| `RAG_MODE_ROUTING_ENABLED` | `false` | Allow classifier verdicts to select FAST or STANDARD plans. |
| `RAG_FAST_ENABLED` | `false` | Allow FAST execution after classifier gates pass. |
| `RAG_DEEP_MODE_ENABLED` | `false` | Allow Deep controller execution from the chat tool. |
| `RAG_DEEP_SUGGEST_ONLY` | `true` | Return `suggested_mode="DEEP"` instead of executing Deep automatically. |
| `RAG_FALLBACK_ENABLED` | current default | Keep or disable the existing LangGraph LLM fallback path during parity checks. |
| `RAG_MODE_TRACE_VERBOSE` | `false` | Include full features and per-rule scoring in trace for eval/debug runs. |

Rollout order:

1. Enable `RAG_MODE_SHADOW_ENABLED` only.
2. Add classifier eval coverage and inspect trace mismatches.
3. Enable `RAG_MODE_ROUTING_ENABLED` for STANDARD-equivalent plans only.
4. Enable `RAG_FAST_ENABLED` after FAST false-positive gates pass.
5. Enable `RAG_DEEP_MODE_ENABLED` with `RAG_DEEP_SUGGEST_ONLY=true`.
6. Disable suggest-only only after Deep answer/citation evaluation passes.

---

## Section 1: Mode Classifier

### 1.1 Modes

| Mode | Purpose | Target P50 | Target P95 | Hot-path LLM | Retrieval calls |
| --- | --- | ---: | ---: | ---: | ---: |
| FAST | Precise single-fact lookup | <600 ms after active routing | <1000 ms | 0 | 1 |
| STANDARD | Default document retrieval | ~1000-1300 ms initially | <1800 ms | 0 | 1 |
| DEEP | Multi-hop, compare, exhaustive, temporal, low-confidence synthesis | 2000-4000 ms | <6000 ms | 1-2 | 2-3 internal calls |

The latency table separates initial engineering targets from ideal future performance. The first production objective is to preserve V3Q FP16 quality and remove unnecessary LLM gates; further latency reductions require evaluation proof.

### 1.2 Core Principles

```text
STANDARD is the safe default.
FAST requires single-entity, single-fact, high-scope confidence.
DEEP intent overrides FAST.
User "quick" may downgrade DEEP to STANDARD, but cannot force FAST on complex queries.
```

### 1.3 QueryFeatures

The original draft proposed a large feature object. Implementation should start smaller and add fields only when tests prove they are needed.

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

Sources:

- Existing QueryPlan output.
- Filename registry and context-file constraints.
- Rule-based keyword patterns.
- Conversation context only for lightweight reference detection.

No LLM is used for feature extraction.

### 1.4 Mode Scoring

```python
FAST_THRESHOLD = 0.75
DEEP_THRESHOLD = 0.40

def score_fast(f: QueryFeatures) -> float:
    if f.entity_count >= 2:
        return 0.0
    if f.has_compare_intent or f.has_exhaustive_intent or f.has_temporal_evolution_intent:
        return 0.0
    if f.has_negation:
        return 0.0
    if f.question_type not in ("fact", "locate", "quote"):
        return 0.0

    score = 0.0
    if f.has_single_file_match and f.has_anchor:
        score += 0.40
    elif f.has_single_file_match:
        score += 0.30
    elif f.has_precise_scope:
        score += 0.25

    if f.entity_count <= 1:
        score += 0.30
    if not f.has_summary_intent and not f.has_causal_intent:
        score += 0.15
    if f.scope_is_global:
        score -= 0.20
    return min(max(score, 0.0), 1.0)


def score_deep(f: QueryFeatures) -> float:
    score = 0.0
    if f.has_compare_intent and f.entity_count >= 2:
        score += 0.50
    if f.has_exhaustive_intent and (f.scope_is_global or f.entity_count >= 2):
        score += 0.45
    if f.has_temporal_evolution_intent:
        score += 0.45
    if f.has_summary_intent and f.entity_count >= 2:
        score += 0.30
    if f.has_summary_intent and f.scope_is_global:
        score += 0.25
    if f.has_causal_intent and (f.entity_count >= 2 or f.scope_is_global):
        score += 0.25
    if f.has_negation and f.has_exhaustive_intent:
        score += 0.20
    return min(score, 1.0)
```

Do not score "version entity only" as DEEP. A query like `v3 默认超时时间是多少` can still be FAST if it is a single scoped fact.

### 1.5 Classification Logic

```python
def classify_mode(features: QueryFeatures) -> ModeVerdict:
    fast_score = score_fast(features)
    deep_score = score_deep(features)

    if features.user_explicit_deep:
        return ModeVerdict("DEEP", "user_explicit_deep", fast_score, deep_score, features)

    if features.is_context_reference:
        return ModeVerdict("STANDARD", "context_reference", fast_score, deep_score, features)

    if deep_score >= DEEP_THRESHOLD:
        if features.user_explicit_fast:
            return ModeVerdict("STANDARD", "fast_requested_but_complex", fast_score, deep_score, features)
        return ModeVerdict("DEEP", "complex_intent", fast_score, deep_score, features)

    if features.user_explicit_fast and fast_score >= 0.50:
        return ModeVerdict("FAST", "user_explicit_fast_simple", fast_score, deep_score, features)

    if fast_score >= FAST_THRESHOLD:
        return ModeVerdict("FAST", "precise_single_fact", fast_score, deep_score, features)

    return ModeVerdict("STANDARD", "default", fast_score, deep_score, features)
```

### 1.6 Shadow-First Requirement

The classifier must ship in shadow mode first:

- It writes `mode_initial`, `mode_reason`, `fast_score`, and `deep_score` into trace.
- It does not affect retrieval behavior until the classifier eval set passes.

Acceptance before active routing:

- At least 100 labeled classifier examples.
- FAST false positives on complex queries below 2%.
- DEEP trigger rate measured on representative queries and below 15%.

Classifier eval artifact:

```text
eval/datasets/rag_mode_classifier_v1.jsonl
```

Each row should be a compact, hand-labeled routing example:

```json
{
  "id": "mode-001",
  "query": "compare A and B cooling plans",
  "context_files": [],
  "expected_mode": "DEEP",
  "complexity_label": "compare",
  "must_not_fast": true,
  "notes": "Two entities and compare intent require non-FAST routing."
}
```

Required label coverage:

- precise single-file facts
- single-file facts with anchors
- global facts without file scope
- single-file summaries
- multi-field extraction from one file
- compare / contrast queries
- exhaustive global queries
- temporal evolution queries
- negation plus exhaustive intent
- context-reference queries such as "what about this one?"
- user-explicit fast on simple and complex queries
- user-explicit deep on simple and complex queries

Classifier reports must include:

- total examples and examples by `expected_mode`
- confusion matrix
- FAST false-positive rate on `must_not_fast=true`
- DEEP trigger rate
- examples where `mode_reason` disagrees with the label rationale
- top 10 highest-risk mismatches with features and scores

### 1.7 Typical Routing Examples

| Query | Entities | Intent | Mode |
| --- | ---: | --- | --- |
| `《A手册》X参数默认值` | 1 | fact | FAST |
| `XJR-500 额定功率` | 1 | fact | FAST |
| `第三章 3.2节 安全阈值` | 1 | fact + anchor | FAST |
| `v3 默认超时时间是多少` | 1 | fact + version entity | FAST |
| `为什么需要设置安全阈值` | 0 | causal single scope | STANDARD |
| `总结《A手册》第三章的内容` | 1 | summary single scope | STANDARD |
| `单表内X型号的功率和电压` | 1 | multi-field extraction | STANDARD |
| `安全相关的所有要求有哪些` | 0 | exhaustive global | DEEP |
| `比较A和B的散热方案` | 2 | compare | DEEP |
| `X标准从v1到v3的变化` | 1 | temporal evolution | DEEP |
| `除了A还有哪些方案` | 1 | negation + exhaustive | DEEP |
| `这个呢？` | 0 | reference | STANDARD |
| `快速比较A和B` | 2 | compare + user fast | STANDARD |

---

## Section 2: RAG Pipeline

### 2.1 RetrievalPlan

Plans are configuration objects over the existing retrieval function. They are not separate pipeline implementations.

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

Initial plans:

| Plan | Retrieval behavior |
| --- | --- |
| FAST | Existing QueryPlan + retrieval + rerank, with smaller CE input only for high-confidence single-file/anchored facts. |
| STANDARD | Quality-first baseline using current V3Q FP16 behavior. No aggressive L1/CE compression by default. |
| DEEP | Not used directly for sub-retrieval. Deep controller runs STANDARD subqueries with `allow_deep=False`. |

### 2.2 Capability Matrix

| Capability | FAST | STANDARD | DEEP |
| --- | --- | --- | --- |
| QueryPlan scope matching | yes | yes | yes |
| dense + sparse retrieval | yes | yes | yes, through STANDARD subqueries |
| hybrid guarantee | yes | yes | yes, through STANDARD subqueries |
| layered rerank | yes, conservative | yes, quality-first | yes, through STANDARD subqueries |
| structure rerank | yes | yes | yes |
| confidence verdict | yes | yes | yes |
| HyDE / step-back LLM generation | no | no | optional, budgeted |
| agent subquery decomposition | no | no | yes, internal controller |
| recursive DEEP | no | no | no |
| Agent tool calls per turn | 1 | 1 | 1 external call, multiple internal retrievals |

### 2.3 Public Retrieval Boundary

Keep `retrieve_documents` as the public retrieval boundary and evolve it in place.

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

Return shape remains compatible:

```python
{
    "docs": [...],
    "meta": {
        ...
    },
}
```

New meta fields:

- `mode_initial`
- `mode_final`
- `mode_reason`
- `fast_score`
- `deep_score`
- `retrieval_plan`
- `answerable`
- `confidence_score`
- `risk_score`
- `confidence_reasons`
- `suggested_mode`
- `needs_clarification`
- `clarification_question`

Legacy meta fields remain until all callers and reports are migrated.

Legacy compatibility rules:

- Keep `fallback_required` while existing reports and graph tests depend on it.
- Map `fallback_required=True` to `answerable=False` only when structured confidence is enabled and risk is at or above the configured not-answerable threshold.
- Map low-risk results to `answerable=True` even when legacy `fallback_required` is absent.
- Do not remove `grade_score`, `grade_route`, `rewrite_needed`, or `fallback_executed` from LangGraph traces until streaming and regression reports are migrated.
- New callers should read `answerable`, `suggested_mode`, and `needs_clarification`; old callers may continue reading `fallback_required`.

### 2.4 Pipeline Orchestration

The pipeline remains a pure, staged function chain inside the existing retrieval boundary:

```text
QueryPlan
  -> Mode classifier
  -> RetrievalPlan
  -> dense/sparse embeddings
  -> existing Milvus retrieval path
  -> L1/CE/structure rerank
  -> structured confidence verdict
  -> docs + meta
```

The first implementation should not use `asyncio.run()` inside retrieval. Current runtime includes sync tools, threadpool execution, and async streaming. If async retrieval is needed later, expose separate sync and async wrappers instead of nesting event loops.

### 2.5 Two-Level Confidence

Pre-CE confidence controls candidate expansion:

| Signal | Weight |
| --- | ---: |
| candidate pool too small | +2.0 |
| entity coverage weak | +2.0 |
| dense/sparse disagree | +1.0 |
| L1 top margin too small | +1.0 |
| precise scope not matched | +2.0 |

Post-CE confidence controls answerability and suggested upgrade:

| Signal | Weight |
| --- | ---: |
| top score too low | +2.0 |
| CE margin too small | +1.5 |
| evidence count too low | +2.0 |
| entity coverage weak | +2.0 |
| answer coverage weak | +2.0 |
| precise scope not matched | +2.0 |
| route disagreement | +1.0 |

Decision:

| Risk | Decision |
| ---: | --- |
| <2.0 | answerable, high confidence |
| 2.0-3.9 | answerable, medium confidence |
| 4.0-5.9 | not answerable, suggested upgrade |
| >=6.0 | not answerable, clarification needed |

### 2.6 ConfidenceVerdict

Extend `backend/rag/confidence.py` in place.

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

- FAST low confidence retries STANDARD immediately.
- STANDARD high risk returns `suggested_mode="DEEP"` to the chat layer.
- Retrieval does not silently run DEEP as a side effect.
- If DEEP is disallowed, return a not-answerable result or clarification.

This closes the ambiguity of simply setting `mode_final="DEEP"` without specifying whether Deep actually executed.

Suggested-mode semantics:

| Current mode | Condition | Result |
| --- | --- | --- |
| FAST | confidence below FAST threshold | Run STANDARD immediately and set `mode_final="STANDARD"`. |
| STANDARD | high risk and Deep disabled | Return `answerable=False`, `suggested_mode="DEEP"`, and a safe explanation. |
| STANDARD | high risk and Deep suggest-only | Return `answerable=False`, `suggested_mode="DEEP"`; do not execute Deep. |
| STANDARD | high risk and Deep active | Execute Deep in chat layer, not inside `retrieve_documents`. |
| DEEP subquery | any low-confidence result | Return partial evidence to Deep coverage evaluator; never recurse. |

### 2.7 RetrievalVerdict Shape

A full dataclass can be introduced later if useful, but phase one should keep the current dict return to avoid broad caller churn. The conceptual verdict is:

```python
@dataclass(frozen=True)
class RetrievalVerdict:
    docs: list[dict]
    mode_initial: str
    mode_final: str
    answerable: bool
    confidence_score: float
    risk_score: float
    suggested_mode: str | None
    needs_clarification: bool
    trace: dict
```

Implementation should prefer dict-compatible meta first. Introduce the dataclass only if it removes repeated conversion or improves tests.

---

## Section 3: Deep Mode Planner-Controller

### 3.1 Architecture

```text
Original query
  -> Mode classifier says DEEP, or STANDARD confidence suggests DEEP
  -> Deep controller
      -> rule decomposer
      -> optional LLM planner fallback
      -> STANDARD sub-retrievals, allow_deep=False
      -> evidence accumulator
      -> coverage evaluator
      -> synthesis with citations
```

Deep Mode is implemented in `backend/chat/deep_mode.py` because it is chat orchestration, not low-level retrieval. It should not be placed in `agent.py`, `tools.py`, or `rag/utils.py`.

Deep Mode returns an answer-ready result, not just another document list. The chat tool should format that result so the outer Agent preserves the cited answer and does not invent additional claims.

### 3.2 Core Data Structures

```python
@dataclass(frozen=True)
class SubQuery:
    id: str
    query: str
    target_entity: str | None
    target_aspect: str | None
    strategy: str
    required: bool = True
    source: str = "rule"


@dataclass
class BudgetTracker:
    max_retrieval_calls: int
    max_llm_calls: int
    max_wall_time_ms: int
    started_at: float = 0.0
    retrieval_calls: int = 0
    llm_calls: int = 0
```

Evidence accumulation starts inside `deep_mode.py`. Do not add `backend/rag/evidence.py` unless the same evidence model is needed outside Deep Mode.

```python
@dataclass
class EvidenceItem:
    dedupe_key: str
    doc: dict
    subquery_ids: set[str]
    target_entities: set[str]
    target_aspects: set[str]
    route_sources: set[str]
    best_score: float
```

```python
@dataclass
class DeepModeResult:
    final_answer: str
    citations: list[dict]
    evidence_items: list[EvidenceItem]
    coverage: dict
    missing_entities: list[str]
    missing_aspects: list[str]
    budgets: dict
    partial: bool
    trace: dict
```

### 3.3 Query Decomposer

Rules first:

1. Multi-entity compare: one subquery per entity/aspect.
2. Temporal evolution: earliest state, latest state, change signal.
3. Exhaustive global query: broad STANDARD retrieval plus optional targeted follow-up if coverage is low.
4. Ambiguous complex query: LLM planner fallback, capped at 3 subqueries.

Subquery invariant:

```python
retrieve_documents(
    sq.query,
    context_files=context_files,
    mode_override="STANDARD",
    allow_deep=False,
)
```

### 3.4 Budget

| Budget | Default |
| --- | ---: |
| Max internal retrieval calls | 3 |
| Max planner LLM calls | 1 |
| Max synthesis LLM calls | 1 |
| Max wall time | 6000 ms |
| Max evidence items sent to synthesis | 15 |
| Max chars per evidence item | 600 |

Budget exhaustion returns partial evidence with explicit missing coverage. It must not fabricate an answer.

### 3.5 Coverage Evaluation

Coverage tracks:

- entities covered / partial / missing
- aspects covered / partial / missing
- evidence count per dimension
- duplicate evidence rate
- whether second and third retrieval calls added new evidence

Early stop:

- overall coverage >= 0.9
- no required entity missing
- no required aspect missing

### 3.6 Synthesizer

The synthesizer receives capped evidence and coverage state.

Prompt requirements:

1. Answer conclusion first.
2. Cite filename and page for each key claim.
3. State missing evidence explicitly.
4. Do not infer across missing dimensions.
5. If comparing A and B but only A has evidence, say B is missing.

Output contract:

- `final_answer` is the answer the user should see.
- Every key claim in `final_answer` must have at least one citation object.
- Each citation includes `filename`, `page_number`, `chunk_id`, and `subquery_id`.
- Missing coverage is part of the answer, not only trace metadata.
- If `partial=true`, the first paragraph must say that the answer is partial.
- If no cited evidence supports the requested comparison or exhaustive answer, return a refusal-to-answer with missing coverage instead of synthesis.

### 3.7 Tool Enhancement

`backend/chat/tools.py` keeps the one-call guard. It may internally dispatch to:

- FAST/STANDARD retrieval formatting.
- Deep controller formatting.

External Agent behavior remains one call:

```text
Agent -> search_knowledge_base(query) -> retrieval or deep result -> final answer
```

Tool output caps:

| Setting | Default |
| --- | ---: |
| Max docs returned to Agent | 5 |
| Max chars per doc | 800 |
| Include diagnostics | yes |
| Store trace for frontend | yes |

Tool response rules:

- FAST/STANDARD returns retrieved chunks plus confidence diagnostics, as today.
- DEEP returns a `Deep Mode Answer` block, citations, missing coverage, and capped evidence snippets.
- The outer Agent prompt should instruct the model to preserve `Deep Mode Answer` claims and citations, not re-synthesize unsupported claims.
- Frontend trace stores both `deep_trace` and the underlying sub-retrieval traces.

---

## Section 4: Observability

### 4.1 Pipeline Metrics

Mode:

- `mode_initial`
- `mode_final`
- `mode_reason`
- `fast_score`
- `deep_score`
- `retrieval_plan`
- `mode_shadow_mismatch`
- `mode_override_used`

Latency:

- `mode_fast_p50_ms`, `mode_fast_p95_ms`
- `mode_standard_p50_ms`, `mode_standard_p95_ms`
- `mode_deep_p50_ms`, `mode_deep_p95_ms`

Retrieval quality:

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

### 4.2 Confidence Metrics

- `answerable`
- `confidence_score`
- `risk_score`
- `confidence_reasons`
- `suggested_mode`
- `needs_clarification`
- `clarification_question`
- `fast_to_standard_upgrade_rate`
- `standard_to_deep_suggestion_rate`

### 4.3 Deep Mode Metrics

- `deep_trigger_rate`
- `deep_trigger_reason_distribution`
- `deep_rule_decompose_rate`
- `deep_llm_planner_rate`
- `deep_avg_retrieval_calls`
- `deep_avg_llm_calls`
- `deep_avg_evidence_count`
- `deep_avg_coverage`
- `deep_timeout_rate`
- `deep_budget_exceeded_rate`
- `deep_second_call_useful_rate`
- `deep_third_call_useful_rate`
- `deep_duplicate_evidence_rate`
- `deep_early_stop_rate`
- `deep_missing_entity_rate`
- `deep_missing_aspect_rate`
- `deep_citation_coverage_rate`
- `deep_partial_answer_rate`
- `deep_unsupported_claim_count`

### 4.4 Agent Decision Metrics

- `tool_call_rate`
- `forced_retrieval_rate`
- `no_retrieval_answer_rate`
- `agent_decision_latency_ms`
- `knowledge_tool_guard_hit_rate`

---

## Implementation Phases

| Phase | Content | Risk | Validation |
| --- | --- | --- | --- |
| P0 | Adopt/document `V3Q + FP16` as quality-first baseline. Behavior otherwise unchanged. | Low | tests + smoke eval |
| P1 | Add shadow Mode Classifier and plan trace. No behavior change. | Low | classifier eval set |
| P2 | Add plan-aware retrieval parameters to existing `retrieve_documents`. STANDARD remains quality-first. | Medium | saved-row regression |
| P3 | Extend confidence into structured verdicts while preserving legacy meta fields. | Medium | fallback/confidence tests |
| P4 | Activate FAST only for high-confidence precise fact queries; retry STANDARD on low confidence. | Medium | full gold eval gates |
| P5 | Add internal Deep controller in chat layer. Subqueries are STANDARD and non-recursive. | Medium-High | deep eval set |
| P6 | Sunset compatibility fallback for FAST/STANDARD after parity is proven; keep rollback flag for one release. | Medium | graph/non-graph parity |

The original draft's intent remains, but implementation is staged to avoid a big-bang rewrite.

P6 note: current code already allows the fallback path to be disabled by configuration. P6 is not a big removal task; it is a controlled cleanup after traces, streaming events, and reports no longer depend on legacy fallback fields.

---

## File Changes

### New Files

Only two new production files are planned initially:

| File | Responsibility | Why new |
| --- | --- | --- |
| `backend/rag/modes.py` | Query feature extraction, mode scoring, mode verdicts, retrieval plans. | Keeps mode policy separate from QueryPlan document-scope parsing. |
| `backend/chat/deep_mode.py` | Deep controller, subquery planning, budget tracking, evidence accumulation, synthesis. | Keeps multi-step chat orchestration out of `agent.py`, `tools.py`, and `rag/utils.py`. |

Do not create generation-labeled modules such as `pipeline_v2.py`, `confidence_v2.py`, or `classifier_v4.py`.

### New Evaluation Artifacts

| File | Responsibility |
| --- | --- |
| `eval/datasets/rag_mode_classifier_v1.jsonl` | Hand-labeled routing examples for classifier gates. |
| `eval/datasets/rag_deep_mode_v1.jsonl` | Multi-hop, compare, exhaustive, and temporal questions with expected cited coverage. |
| `scripts/rag_eval/evaluate_mode_classifier.py` | Focused classifier report without Milvus or CrossEncoder dependency. |
| `scripts/rag_eval/evaluate_deep_mode.py` | Deep answer/citation coverage report. |

### Modified Files

| File | Changes |
| --- | --- |
| `backend/rag/utils.py` | Accept optional mode controls, resolve retrieval plans, add mode/plan/confidence meta. |
| `backend/rag/confidence.py` | Add structured confidence verdict while preserving legacy fields. |
| `backend/rag/layered_rerank.py` | Allow plan-aware L1/L2/L3 parameters where needed. |
| `backend/rag/pipeline.py` | Keep compatibility path; later bypass LLM grading/routing for FAST/STANDARD when confidence verdict is active. |
| `backend/chat/tools.py` | Keep one-call guard, cap output, dispatch internally to Deep controller when needed. |
| `backend/chat/agent.py` | Update prompt and streaming trace handling for mode-aware retrieval. |
| `backend/config.py` | Add flags and defaults. |
| `scripts/rag_eval/variants.py` | Add shadow/active mode experiment variants without changing existing variants. |
| `tests/*` | Add classifier, plan, confidence, FAST retry, Deep controller, and tool guard tests. |

### Preserved Files

| File | Note |
| --- | --- |
| `backend/rag/query_plan.py` | Reused as document-scope parser, not expanded into mode policy. |
| `backend/rag/retrieval.py` | Reuse dense/sparse/RRF helpers. |
| `backend/rag/rerank.py` | Reuse CE and score fusion. |
| `backend/rag/context.py` | Reuse auto-merge and structure rerank. |
| `backend/rag/diagnostics.py` | Reuse failure classification. |

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
| Classifier eval examples | at least 100, with all required label categories covered |
| Classifier confusion matrix | no FAST prediction for hard-blocked complex categories |
| Deep citation coverage | 100% of key claims cite at least one evidence item |
| Deep unsupported claims | 0 on deep eval set |
| Deep partial-answer honesty | 100% of partial answers state missing coverage |
| Error rate | 0.0 on eval runs |
| Recursive DEEP calls | 0 |

Latency gates:

| Mode | Gate |
| --- | --- |
| FAST | P50 under 600 ms after active routing |
| STANDARD | P50 around 1000-1300 ms initially, P95 under 1800 ms |
| DEEP | P95 under 6000 ms |

If a faster plan misses quality gates, it remains experimental.

---

## Test Plan

Unit tests:

- Mode classifier examples.
- FAST hard blocks.
- User explicit fast/deep handling.
- Context reference behavior.
- Confidence verdict risk scoring.
- FAST low-confidence retry to STANDARD.
- Deep subqueries use STANDARD with `allow_deep=False`.
- Tool guard still prevents repeated Agent tool calls.
- Tool output caps docs and characters.

Integration tests:

- `search_knowledge_base` returns legacy-compatible text plus diagnostics.
- Streaming still emits RAG steps.
- Attached context files still constrain retrieval.
- Graph fallback path remains available.
- Deep tool formatting preserves final answer, citations, and missing coverage.
- Existing Agent tool guard still counts Deep as one external knowledge-base call.

Verification commands:

```powershell
uv run pytest tests/ -q
uv run ruff check backend/ scripts/ tests/
uv run python -m compileall backend scripts
uv run python scripts/evaluate_rag_matrix.py --dataset eval/datasets/rag_doc_gold.jsonl --variants V3Q --mode retrieval --skip-reindex --run-id baseline-check
```

Focused mode-routing evaluation:

```powershell
uv run python scripts/evaluate_rag_matrix.py --dataset eval/datasets/rag_doc_gold.jsonl --variants V4_SHADOW,V4_ACTIVE --mode retrieval --skip-reindex --run-id mode-routing-check
uv run python scripts/rag_eval/evaluate_mode_classifier.py --dataset eval/datasets/rag_mode_classifier_v1.jsonl
uv run python scripts/rag_eval/evaluate_deep_mode.py --dataset eval/datasets/rag_deep_mode_v1.jsonl --skip-reindex
```

---

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| FAST routes complex query | Low recall or wrong answer | Strict hard blocks, shadow eval, immediate STANDARD retry on low confidence. |
| STANDARD over-compresses candidates | Regression like EXP_C5 | Keep V3Q FP16 quality-first baseline; require full gold gates for compression. |
| DEEP over-triggers | Latency and cost spike | Trigger cap, metrics, explicit thresholds, traffic sampling. |
| DEEP recurses | Budget explosion | `allow_deep=False` on all subqueries and a dedicated invariant test. |
| Agent loops tools | Slow or unstable answers | Preserve one-call guard; Deep runs internally. |
| Confidence miscalibration | Bad answerability decisions | Saved-row regression plus confidence calibration set. |
| Too many new files | Codebase drift | Start with only `modes.py` and `deep_mode.py`; extract later only with evidence. |
| LangGraph removal breaks streaming trace | UX regression | Keep compatibility switch and shadow compare before removal. |
| Versioned module sprawl | Duplicate logic and cleanup cost | No generation labels in production module names. |
| Deep answer is re-synthesized by outer Agent | Citations drift or unsupported claims appear | Return an answer-ready Deep block and update Agent prompt to preserve it. |
| Flag combinations create unclear behavior | Hard-to-debug rollout failures | Keep a single rollout table and test each flag combination that changes control flow. |
| Classifier eval set is too easy | FAST looks safer than it is | Include hard negatives and user-explicit fast on complex queries. |

---

## Open Decisions

1. FAST remains shadow-only until classifier evaluation exists.
2. STANDARD starts from V3Q FP16 quality-first behavior.
3. Deep synthesis prompt should be citation-first and must expose missing evidence.
4. Candidate compression requires full gold evaluation before default activation.
5. Any future module extraction must remove complexity from an existing file, not create a parallel implementation stack.
6. Deep execution should stay suggest-only until the Deep answer eval set exists.

---

## Final Design Position

The original design direction is retained: pipeline-first retrieval, mode-aware routing, and bounded agentic Deep Mode.

The implementation strategy is tightened:

- Reuse current retrieval and evaluation assets.
- Add only essential boundaries.
- Avoid versioned production module names.
- Prove routing before enabling it.
- Keep Agent behavior bounded to one knowledge-base tool call.
- Let full evaluation gates decide when a faster plan becomes default.
