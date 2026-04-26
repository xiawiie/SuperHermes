# V3F Rebuild Decision

- Status: excluded_from_baseline
- Reason: v3f_rebuild_blocked_or_unstable

## V3F
- File@5: 0.416
- FileCandRecall: 0.44
- P50/P95: 33.97509999922477 / 3047.0336999977008
- Error: 0.0

## GS3 Reference
- File@5: 0.92
- FileCandRecall: 1.0
- P50/P95: 1144.175900000846 / 1609.5243599935202
- Error: 0.0

## Gates
- FileCandRecall >= 0.95
- File@5 >= 0.85
- ErrorRate == 0
- P50 < GS3*0.75

## Notes
- Multiple rebuild attempts stalled in Milvus reindex stage.
- Exclude V3F from v3.1 baseline claims.
