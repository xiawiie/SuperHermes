# RAG Matrix Evaluation Summary

Generated at: `2026-04-26T13:21:58`
Rows: `1000`

## Build Info

- Git commit: `2aaf11ba89e4447bab8d11a78ca44e09ece38e87`
- Corpus path: `None`
- Dataset SHA256: `None`
- Index build timestamp: `not rebuilt in this run`

## Variant Metrics

| Variant | Rows | File@5 | File+Page@5 | FileCandRecall | HardNeg@5 | Anchor@5 | MRR | Chunk@5 | Root@5 | P50 ms | P95 ms | Error | FallbackReq | FallbackExec | Helped | Hurt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GB0 | 125 | 0.744 | 0.504 | 0.952 | 0.027 | 0.256 | 0.296 | 0.000 | 0.529 | 939.093 | 1060.973 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| GS1 | 125 | 0.744 | 0.504 | 0.952 | 0.027 | 0.256 | 0.296 | 0.000 | 0.529 | 822.565 | 1037.841 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| GS2 | 125 | 0.872 | 0.680 | 1.000 | 0.006 | 0.344 | 0.304 | 0.000 | 0.644 | 847.220 | 1058.026 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| GS2H | 125 | 0.864 | 0.616 | 1.000 | 0.006 | 0.336 | 0.300 | 0.000 | 0.609 | 835.783 | 930.334 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| GS2HR | 125 | 0.992 | 0.752 | 1.000 | 0.005 | 0.376 | 0.304 | 0.000 | 0.770 | 1531.661 | 2059.967 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| GS3 | 125 | 0.920 | 0.712 | 1.000 | 0.006 | 0.368 | 0.708 | 0.655 | 0.724 | 1144.176 | 1609.524 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| V3F | 125 | 0.416 | 0.280 | 0.440 | 0.015 | 0.168 | 0.282 | 0.276 | 0.310 | 33.975 | 3047.034 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| V3Q | 125 | 0.992 | 0.768 | 1.000 | 0.008 | 0.368 | 0.704 | 0.724 | 0.793 | 3938.231 | 4450.329 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Qrel Coverage

| Variant | FileQrel | PageQrel | ChunkQrel | RootQrel |
| --- | ---: | ---: | ---: | ---: |
| GB0 | 1.000 | 1.000 | 0.696 | 0.696 |
| GS1 | 1.000 | 1.000 | 0.696 | 0.696 |
| GS2 | 1.000 | 1.000 | 0.696 | 0.696 |
| GS2H | 1.000 | 1.000 | 0.696 | 0.696 |
| GS2HR | 1.000 | 1.000 | 0.696 | 0.696 |
| GS3 | 1.000 | 1.000 | 0.696 | 0.696 |
| V3F | 1.000 | 1.000 | 0.696 | 0.696 |
| V3Q | 1.000 | 1.000 | 0.696 | 0.696 |

## Experimental Evidence Metrics

| Variant | ChunkRows | RootRows | ChunkMRR | RootMRR | AnswerSupport@5 exp |
| --- | ---: | ---: | ---: | ---: | ---: |
| GB0 | 87 | 87 | 0.000 | 0.392 | 0.517 |
| GS1 | 87 | 87 | 0.000 | 0.392 | 0.517 |
| GS2 | 87 | 87 | 0.000 | 0.508 | 0.632 |
| GS2H | 87 | 87 | 0.000 | 0.480 | 0.598 |
| GS2HR | 87 | 87 | 0.000 | 0.645 | 0.759 |
| GS3 | 87 | 87 | 0.580 | 0.637 | 0.713 |
| V3F | 87 | 87 | 0.210 | 0.228 | 0.310 |
| V3Q | 87 | 87 | 0.575 | 0.635 | 0.793 |

## Paired Comparisons

- `GS1_vs_GB0`: wins=0, losses=0, ties=125, missing=0
- `GS2_vs_GS1`: wins=1, losses=0, ties=124, missing=0
- `GS2H_vs_GS2`: wins=0, losses=1, ties=124, missing=0
- `GS2HR_vs_GS2H`: wins=1, losses=0, ties=124, missing=0
- `GS3_vs_GS2HR`: wins=57, losses=0, ties=68, missing=0
- `V3Q_vs_GS3`: wins=12, losses=13, ties=100, missing=0
- `V3F_vs_V3Q`: wins=1, losses=65, ties=59, missing=0

## Diagnostics

- `GB0`: file_recall_miss=6, ok=63, page_miss=30, ranking_miss=26
- `GS1`: file_recall_miss=6, ok=63, page_miss=30, ranking_miss=26
- `GS2`: ok=85, page_miss=24, ranking_miss=16
- `GS2H`: ok=77, page_miss=31, ranking_miss=17
- `GS2HR`: ok=94, page_miss=30, ranking_miss=1
- `GS3`: ok=89, page_miss=26, ranking_miss=10
- `V3F`: file_recall_miss=70, ok=35, page_miss=17, ranking_miss=3
- `V3Q`: ok=96, page_miss=28, ranking_miss=1

## Rewrite Strategies

- `GB0`: none=125
- `GS1`: none=125
- `GS2`: none=125
- `GS2H`: none=125
- `GS2HR`: none=125
- `GS3`: none=125
- `V3F`: none=125
- `V3Q`: none=125

## Notes

- `FileCandRecall` is strict filename-level recall before rerank; `weak_any_candidate_recall_before_rerank` remains in JSON for legacy diagnostics.
- Primary metrics: `File@5`, `File+Page@5`, `FileCandRecall`, `Chunk@5`, `Root@5`, `HardNeg@5`, `Anchor@5`, `MRR`, `P50/P95`.
- `HardNeg@5` is only meaningful on the gold dataset (hard_negative_files defined); shown as `-` on natural/frozen.
- DeepEval/Ragas/TruLens are not runtime dependencies in this report.
