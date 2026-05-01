# RAG Candidate Strategy Contract

## Status

Runtime contract: active.

RAG now keeps L0-L3 as the global pipeline stage model:

```text
L0 Retrieval
L1 Candidate Selection
L2 Shared Rerank
L3 Shared Postprocess
```

`standard` and `layered` are candidate strategies only. They may choose different L0/L1 candidate pools, but every complete retrieval result must go through `finish_retrieval_pipeline()`.

## Configuration

The runtime candidate strategy is selected by one environment variable:

```text
RAG_CANDIDATE_STRATEGY=standard | layered
```

Default is `standard`. Invalid values fall back to `standard` and are recorded as trace warnings/stage errors instead of failing startup.

Layered internal L0/L1 values are code presets, not production environment knobs. CrossEncoder size, cache, score fusion, structure postprocess, and confidence gate remain shared runtime settings.

## Runtime Fields

Every retrieval trace should include:

```text
pipeline_stage_model = rag-l0-l3-v1
candidate_strategy_requested
candidate_strategy_effective
candidate_strategy_version = candidate-strategy-v2
candidate_strategy_detail
candidate_strategy_fallback_from
rerank_contract = shared_rerank
rerank_contract_version = shared-rerank-v2
rerank_execution_mode
postprocess_contract = shared_retrieval_postprocess
postprocess_contract_version = shared-postprocess-v1
```

`rerank_execution_mode` is one of:

```text
executed
disabled
skipped_candidate_only
failed_before_rerank
failed_with_ranked_candidates
```

## Boundaries

Allowed:

- Add or tune candidate generation behavior inside L0/L1.
- Compare `standard` and `layered` as retrieval candidate strategies.
- Use shared CrossEncoder, cache, fusion, structure postprocess, confidence, and trace fields.

Not allowed:

- Add a second final rerank path inside a candidate strategy.
- Bypass `finish_retrieval_pipeline()` for complete retrieval.
- Add another matrix of layered production environment variables.
- Use one trace field to mean both candidate generation and final rerank behavior.

## Acceptance

- `standard` and `layered` both emit the same L0-L3 trace contract.
- Candidate-only retrieval explicitly marks `skipped_candidate_only`.
- Layered complete retrieval calls the shared rerank path once.
- Eval reports requested/effective candidate strategy and rerank execution mode.
