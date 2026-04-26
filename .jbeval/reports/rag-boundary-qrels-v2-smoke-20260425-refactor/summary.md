# RAG Matrix Evaluation Summary

Generated at: `2026-04-26T00:17:57`
Rows: `20`

## Build Info

- Git commit: `4183268294355daa0eeb850e705bfb96a3886969`
- Corpus path: `C:\Users\goahe\Desktop\Project\doc`
- Dataset SHA256: `a1549cdbd712903a53a9a44c274dc1cb5a1de6cff71d7ac2c6f9ab9996c39b30`
- Index build timestamp: `not rebuilt in this run`

## Coverage Preflight

- Corpus file coverage: 60/60 (1.000) at `C:\Users\goahe\Desktop\Project\doc`
- External qrels: `.jbeval\datasets\rag_chunk_gold_v2.jsonl` matched=125/125, chunk_mode=canonical, root_mode=canonical, conflicts=0, review=0.000
- Qrel coverage: file=1.000, page=1.000, chunk=0.696 (87/125), root=0.696 (87/125)

## Variant Metrics

| Variant | Rows | File@5 | File+Page@5 | FileCandRecall | HardNeg@5 | Anchor@5 | MRR | Chunk@5 | Root@5 | P50 ms | P95 ms | Error | FallbackReq | FallbackExec | Helped | Hurt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GS3 | 10 | 1.000 | 0.700 | 1.000 | 0.040 | 0.400 | 0.800 | 0.750 | 0.750 | 1049.658 | 16766.068 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| V3Q | 10 | 1.000 | 0.700 | 1.000 | 0.040 | 0.400 | 0.670 | 0.750 | 0.750 | 4002.353 | 19813.693 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Qrel Coverage

| Variant | FileQrel | PageQrel | ChunkQrel | RootQrel |
| --- | ---: | ---: | ---: | ---: |
| GS3 | 1.000 | 1.000 | 0.800 | 0.800 |
| V3Q | 1.000 | 1.000 | 0.800 | 0.800 |

## Experimental Evidence Metrics

| Variant | ChunkRows | RootRows | ChunkMRR | RootMRR | AnswerSupport@5 exp |
| --- | ---: | ---: | ---: | ---: | ---: |
| GS3 | 8 | 8 | 0.750 | 0.750 | 0.750 |
| V3Q | 8 | 8 | 0.588 | 0.688 | 0.750 |

## Paired Comparisons

- `V3Q_vs_GS3`: wins=0, losses=2, ties=8, missing=0

## Diagnostics

- `GS3`: ok=7, page_miss=3
- `V3Q`: ok=7, page_miss=3

## Rewrite Strategies

- `GS3`: none=10
- `V3Q`: none=10

## Notes

- `FileCandRecall` is strict filename-level recall before rerank; `weak_any_candidate_recall_before_rerank` remains in JSON for legacy diagnostics.
- Primary metrics: `File@5`, `File+Page@5`, `FileCandRecall`, `Chunk@5`, `Root@5`, `HardNeg@5`, `Anchor@5`, `MRR`, `P50/P95`.
- `HardNeg@5` is only meaningful on the gold dataset (hard_negative_files defined); shown as `-` on natural/frozen.
- DeepEval/Ragas/TruLens are not runtime dependencies in this report.
