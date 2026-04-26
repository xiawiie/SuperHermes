# RAG v3.1 Full Gold Qrel Evaluation Analysis

Date: 2026-04-26

## Artifacts

| Artifact | Path |
| --- | --- |
| Summary | `.jbeval/reports/rag-v3.1-full-gold-qrels-refactor-20260426/summary.md` |
| JSON summary | `.jbeval/reports/rag-v3.1-full-gold-qrels-refactor-20260426/summary.json` |
| Coverage preflight | `.jbeval/reports/rag-v3.1-full-gold-qrels-refactor-20260426/coverage_preflight.json` |
| Results rows | `.jbeval/reports/rag-v3.1-full-gold-qrels-refactor-20260426/results.jsonl` |
| Miss analysis | `.jbeval/reports/rag-v3.1-full-gold-qrels-refactor-20260426/miss_analysis.jsonl` |

## Result

The full qrel-aware run completed successfully: 8 variants, 125 samples each, 1000 rows total, and zero runtime errors. Corpus and collection coverage are 60/60 expected files for every evaluated collection.

Chunk/Root qrels are now active with 87/125 rows covered. These labels are automatic drafts, so they are suitable for engineering diagnosis but not yet final acceptance gates.

## Key Metrics

| Variant | File@5 | File+Page@5 | Chunk@5 | Root@5 | AnswerSupport@5 exp | P50 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GB0 | 0.744 | 0.504 | 0.000 | 0.529 | 0.517 | 939 | 1061 |
| GS1 | 0.744 | 0.504 | 0.000 | 0.529 | 0.517 | 823 | 1038 |
| GS2 | 0.872 | 0.680 | 0.000 | 0.644 | 0.632 | 847 | 1058 |
| GS2H | 0.864 | 0.616 | 0.000 | 0.609 | 0.598 | 836 | 930 |
| GS2HR | 0.992 | 0.752 | 0.000 | 0.770 | 0.759 | 1532 | 2060 |
| GS3 | 0.920 | 0.712 | 0.655 | 0.724 | 0.713 | 1144 | 1610 |
| V3Q | 0.992 | 0.768 | 0.724 | 0.793 | 0.793 | 3938 | 4450 |
| V3F | 0.416 | 0.280 | 0.276 | 0.310 | 0.310 | 34 | 3047 |

## Conclusions

1. `V3Q` is the current quality ceiling, especially for File/Page/Chunk/Root metrics, but its latency makes it unsuitable as the default online path.
2. `GS3` is the best current default baseline because it gives strong Chunk@5 and good latency.
3. `GS2` validates that scoped QueryPlan is useful, but chunk-level evidence remains poor without filename-aware/canonical-compatible indexing.
4. `GS2HR` has high file/page/root scores but `Chunk@5=0.000`, which indicates a qrel/profile granularity mismatch or evidence ID incompatibility.
5. `V3F` should be treated as broken until its profile/index is rebuilt and traced.

## Recommended Next Steps

1. Human-review qrel rows before making Chunk@5 a hard gate.
2. Rebuild `v3_fast` from canonical corpus and rerun only `V3F`.
3. Add page-rank-before/after-rerank diagnostics.
4. Use `GS3` as the default baseline and `V3Q` as quality fallback candidate.
5. Run page-aware fusion ablation after qrel review.
