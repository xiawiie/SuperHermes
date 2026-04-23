# RAG Evaluation Matrix Executor Design

## Purpose

Implement the smallest practical evaluation executor needed to prove whether the current RAG changes improve SuperHermes retrieval quality.

The executor must operationalize the existing documents:

- `RAG_HIERARCHICAL_V1_DESIGN.md`
- `RAG_HIERARCHICAL_V1_EXPERIMENT_MATRIX.md`
- `RAG_DIAGNOSTIC_SPEC.md`

The goal is not to integrate DeepEval, Ragas, or TruLens as dependencies in this round. Their metrics shape the design, but this implementation keeps evaluation deterministic, local, and low-risk.

## Scope

In scope:

- Add a script that runs matrix-style retrieval evaluations.
- Consume `.jbeval/datasets/rag_tuning_derived.jsonl`.
- Output `results.jsonl`, `summary.json`, `summary.md`, and `config.json`.
- Compute retrieval-first metrics:
  - `root_hit@5`
  - `anchor_hit@5`
  - `keyword_hit@5`
  - `MRR`
  - `context_precision_id@5`
  - `irrelevant_context_ratio@5`
  - latency and error rate
- Run `classify_failure()` for each sample where trace and expected fields are available.
- Produce paired comparison counts for A1 vs A0, B1 vs A1, and gate variants where comparable.
- Run the documented matrix using destructive reindex where required.

Out of scope:

- Adding DeepEval, Ragas, TruLens, or any new dependency.
- Full LLM-as-judge E2E scoring.
- Frontend changes.
- Prompt or model-layer automatic diagnosis.
- A production staging/swap indexing system.

## Recommended Approach

Use a new script:

```text
scripts/evaluate_rag_matrix.py
```

The script owns evaluation orchestration and reporting. It should call the existing retrieval pipeline directly rather than adding new API endpoints.

This keeps the change small and avoids changing normal application behavior. The RAG runtime remains responsible for retrieval, rerank, structure rerank, trace emission, and confidence gate decisions. The new script only configures runs, invokes retrieval, measures results, and writes reports.

## Experiment Groups

The executor supports these run variants:

| ID | Reindex mode | Structure rerank | Confidence gate | Purpose |
| --- | --- | --- | --- | --- |
| A0 | `raw` | off | off | Raw text baseline |
| A1 | `title_context` | off | off | Test title-injected retrieval text |
| B1 | `title_context` | on | off | Test structure rerank |
| G0 | reuse B1 index | on | off | Gate baseline |
| G1 | reuse B1 index | on | current thresholds | Recommended gate |
| G2 | reuse B1 index | on | looser thresholds | Check under-triggering tradeoff |
| G3 | reuse B1 index | on | stricter thresholds | Check over-triggering tradeoff |

A0 and A1 require reindex because `EVAL_RETRIEVAL_TEXT_MODE` affects indexed text. B1 and G variants reuse the `title_context` index.

## Data Flow

1. Ensure `.jbeval/datasets/rag_tuning_derived.jsonl` exists.
2. For A0:
   - set `EVAL_RETRIEVAL_TEXT_MODE=raw`
   - run `scripts/reindex_knowledge_base.py`
   - run retrieval evaluation with structure rerank and gate disabled
3. For A1:
   - set `EVAL_RETRIEVAL_TEXT_MODE=title_context`
   - run `scripts/reindex_knowledge_base.py`
   - run retrieval evaluation with structure rerank and gate disabled
4. For B1:
   - reuse A1 index
   - enable structure rerank
   - keep gate disabled
5. For G0/G1/G2/G3:
   - reuse B1 index
   - vary confidence gate thresholds
6. Write one report directory under `.jbeval/reports/<run_id>/`.

## Reporting Contract

Each `results.jsonl` row must include:

- `sample_id`
- `variant`
- `query`
- expected fields used for evaluation
- retrieved chunk summaries
- trace summary
- retrieval metrics
- `diagnostic_result`
- latency and error fields

Each `summary.json` must include:

- run metadata
- per-variant aggregate metrics
- paired comparisons
- diagnostic category counts
- config snapshot

Each `summary.md` must be human-readable and focus on:

- whether A1 beats A0
- whether B1 beats A1
- whether G1 is more balanced than G2/G3
- highest-frequency failure categories
- remaining risks

Each `config.json` must include:

- timestamp
- dataset path
- git status summary
- matrix variants
- environment variables that affect retrieval behavior
- whether destructive reindex was run

## Metric Definitions

`root_hit@5` is true when any top-5 chunk matches an expected root id.

`anchor_hit@5` is true when any top-5 chunk text, section path, section title, or anchor id contains an expected anchor.

`keyword_hit@5` is true when all or enough expected keywords appear in the top-5 text. The first implementation may use "any expected keyword" for compatibility, but the summary must make that explicit.

`MRR` is the reciprocal rank of the first top-5 item matching root, anchor, keyword, or mapped chunk id.

`context_precision_id@5` is the proportion of top-5 chunks that match expected root, anchor, keyword, or mapped chunk id. This follows the spirit of Ragas ID-based context precision without importing Ragas.

`irrelevant_context_ratio@5` is `1 - context_precision_id@5`.

Paired comparison should classify each sample as:

- `win`: new variant improves first relevant rank or changes miss to hit
- `loss`: new variant worsens first relevant rank or changes hit to miss
- `tie`: no rank/hit change

## Error Handling

If retrieval fails for one sample, the executor should record an error row and continue.

If reindex fails, the executor should stop because downstream results would be invalid.

If expected fields are missing, metrics should fall back in this order:

1. `expected_root_ids`
2. `expected_anchors`
3. `expected_keywords`
4. `legacy_gold_chunk_ids`

If none exist, the row should be marked `unscorable`.

## Testing Plan

Add focused unit tests for metric helpers and paired comparison. Avoid tests that require live Milvus or external APIs.

Run:

```text
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

After implementation, run a small smoke evaluation first if the script supports `--limit`, then run the full matrix.

## Self-Review

No placeholder requirements remain.

The design is intentionally bounded to deterministic retrieval evaluation. It does not contradict the existing RAG documents: framework-based judge metrics remain future work, while this implementation makes the documented A/B/G matrix executable now.

The only destructive action is reindexing, which the user explicitly approved.
