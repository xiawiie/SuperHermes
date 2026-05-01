# Historical Layered Candidate Plan

## Obsolete Runtime Contract

This file is historical. It captured an early design where the layered experiment owned more of the retrieval pipeline. That design is no longer the runtime contract.

Current runtime contract:

```text
standard/layered candidate strategy
  -> L0 Retrieval
  -> L1 Candidate Selection
  -> L2 Shared Rerank
  -> L3 Shared Postprocess
```

The current module is `backend/rag/layered_candidates.py`, and the current test file is `tests/test_layered_candidates.py`.

Candidate strategy is selected with:

```text
RAG_CANDIDATE_STRATEGY=standard | layered
```

Historical details from the original experiment were intentionally removed from this plan file so that `docs/superpowers/plans` does not read like active runtime guidance. Use git history for the original task breakdown if needed.
