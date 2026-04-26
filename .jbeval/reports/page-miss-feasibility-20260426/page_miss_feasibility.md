# Page Miss Feasibility Analysis

- Variant: `GS3`
- Rows: `125`
- Page miss rows: `36`
- Target ratio: `0.0278`
- GS3P feasible: `False`

## Buckets

| Bucket | Count |
| --- | ---: |
| file_hit_page_missing | 25 |
| other | 10 |
| root_hit_chunk_wrong | 1 |

## Rule

- page_hit_before_rerank_dropped + root_hit_chunk_wrong >= 50% of page_miss
