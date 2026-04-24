# RAG Main Training Dataset Design

## Purpose

Turn `.jbeval/datasets/rag_doc_gold.jsonl` from a page-level RAG benchmark into a maintainable main training dataset family for SuperHermes.

The current file is valuable, but its present role is evaluation-first. It has clean page/file gold labels, source excerpts, quality checks, and qrels-like positive contexts, but it does not yet provide the chunk-level positives, hard negatives, multi-hop coverage, or answer variants needed for long-term retriever, reranker, and grounded-generation training.

This design keeps the existing benchmark intact and builds derived datasets around it. The main rule is: do not overwrite the current gold file; produce versioned derivative artifacts with clear split, schema, and provenance.

## Current Facts

Observed from `.jbeval/datasets/rag_doc_gold.jsonl`:

- Rows: 125
- Unique source files: 60
- `gold_chunk_ids`: empty for all 125 rows
- `negative_contexts`: empty for all 125 rows
- `positive_contexts`: exactly 1 per row
- `difficulty`: all `single-hop`
- `generation_method`: all `deterministic_source_excerpt_v1`
- `benchmark_ready`: true for all rows
- `answer_type`: `operation` 45, `specification` 42, `troubleshooting` 36, `safety` 2
- `hard_negative_files`: already present for most rows and should be reused as negative-mining hints

The repo already has RAG evaluation infrastructure and hierarchical chunk metadata:

- `backend/document_loader.py` emits `chunk_id`, `root_chunk_id`, `parent_chunk_id`, `anchor_id`, `section_title`, and `retrieval_text`.
- `backend/rag_diagnostics.py` can classify failures using chunk/root/anchor/keyword evidence.
- `scripts/evaluate_rag_matrix.py` already reports retrieval-first metrics.
- `.jbeval/datasets/rag_dataset_quality_review.md` documents why the current v2 set is benchmark-ready but still missing real chunk/root id mapping.

## Design Principle

The dataset should become a family of artifacts, not a single overloaded JSONL.

Evaluation data, retriever/reranker training data, contrastive pairs, query variants, and generation targets need different stability rules. Mixing them into one mutable file will cause leakage and make improvements hard to trust.

The intended asset hierarchy is:

```text
.jbeval/datasets/rag_doc_gold.jsonl
  canonical page-level seed benchmark; read-only input

.jbeval/datasets/rag_doc_splits_v1.json
  file-grouped train/dev/test split manifest

.jbeval/datasets/rag_doc_frozen_eval_v1.jsonl
  frozen dev/test benchmark rows derived by source-file split

.jbeval/datasets/rag_chunk_gold_v1.jsonl
  chunk/root/anchor-aligned training and evaluation rows

.jbeval/datasets/rag_contrastive_train_v1.jsonl
  positive and negative chunk pairs for retriever/reranker training

.jbeval/datasets/rag_generation_train_v1.jsonl
  extractive plus abstractive answers for grounded answer generation
```

## Scope

In scope for the first implementation round:

- Create file-grouped train/dev/test split manifest.
- Derive frozen eval rows without changing source records.
- Align page-level gold to current chunk ids.
- Populate `gold_chunk_ids`, `expected_root_ids`, and `supporting_chunks`.
- Mine negative chunks using existing `hard_negative_files`, same-document wrong pages, and same-domain wrong documents.
- Add dataset quality reports and validation checks.
- Extend evaluation scripts only as needed to consume chunk/root/negative fields.

Out of scope for the first implementation round:

- Training a model.
- Adding new dependencies.
- LLM-generated query expansion at scale.
- Multi-hop and cross-document synthesis.
- Manual annotation UI.
- Replacing the existing page-level benchmark metrics.

## Split Strategy

Split by source file, not by question.

The current 125 rows cover 60 source files, usually 1-3 samples per file. Random sample-level splitting would leak document wording across train/dev/test and inflate results. A split manifest should assign every `gold_files[0]` to exactly one split.

Recommended split:

- `test`: 10-12 source files
- `dev`: 8-10 source files
- `train`: all remaining source files

Balancing constraints:

- Keep answer types represented in each split when possible.
- Keep high-frequency domains such as device manuals represented in all splits.
- Do not split rows from the same source file across train/dev/test.
- Keep the split manifest stable once committed.

The split manifest should contain:

```json
{
  "schema_version": "rag-doc-splits-v1",
  "source_dataset": ".jbeval/datasets/rag_doc_gold.jsonl",
  "split_key": "gold_files[0]",
  "splits": {
    "train": ["...pdf"],
    "dev": ["...pdf"],
    "test": ["...pdf"]
  }
}
```

## Chunk Gold Alignment

Chunk alignment is the most important upgrade. The current rows answer "did retrieval find the right file/page?" but training needs "which chunk is the positive evidence?"

The alignment script should use several signals in order:

1. Exact loose-text match from `source_excerpt` to chunk `text` or `retrieval_text`.
2. Windowed fuzzy match when the excerpt crosses chunk boundaries.
3. Page-constrained match using `gold_files` and `gold_pages`.
4. Anchor/heading match using `expected_anchors`, `source_heading`, `anchor_id`, `section_title`, and `section_path`.
5. Keyword coverage using `expected_keywords`.

Only chunks from the same source file and gold page should be eligible for exact gold assignment. Cross-page chunks may be allowed only when the chunk metadata explicitly spans the gold page via `page_start` and `page_end`.

Each aligned row should add:

```json
{
  "gold_chunk_ids": ["..."],
  "expected_root_ids": ["..."],
  "supporting_chunks": [
    {
      "chunk_id": "...",
      "root_chunk_id": "...",
      "parent_chunk_id": "...",
      "file_name": "...",
      "page_number": 12,
      "page_start": 12,
      "page_end": 12,
      "anchor_id": "1.2",
      "section_title": "...",
      "anchor_text": "...",
      "match_method": "exact|window|anchor_keyword|manual",
      "match_score": 0.0,
      "relevance": 3
    }
  ]
}
```

Rows that cannot be aligned should not be silently dropped. Mark them as `alignment_status: "failed"` and include failure reasons such as `missing_source_page`, `excerpt_not_found`, `ambiguous_match`, or `index_not_available`.

## Negative Mining

The current dataset has `hard_negative_files` but not `negative_contexts`. That is the fastest path to usable contrastive data.

Each training row should target:

- Positives: 1-3 chunks
- Hard negatives: 2-4 chunks
- Easy negatives: 1-2 chunks

Negative sources, in priority order:

1. Same source file, wrong page or wrong section.
2. Existing `hard_negative_files` with similar product/domain labels.
3. Same answer type but different source file.
4. Lexically similar chunks retrieved by baseline search that do not overlap gold file/page/chunk ids.
5. Random same-language chunks for easy negatives.

Hard negative eligibility rules:

- Must not be in `gold_chunk_ids`.
- Must not share a `root_chunk_id` with a positive unless explicitly marked as `near_miss_same_root`.
- Must not contain the full `source_excerpt`.
- Should share at least one keyword, domain tag, model name, or answer type with the query.

Negative context shape:

```json
{
  "chunk_id": "...",
  "root_chunk_id": "...",
  "file_name": "...",
  "page_number": 30,
  "text": "...",
  "negative_type": "same_doc_wrong_page|hard_negative_file|same_type_wrong_doc|bm25_near_miss|random_easy",
  "reason": "shares keyword but answers a different operation",
  "relevance": 0
}
```

## Target Schemas

### Chunk Gold Row

```json
{
  "id": "doc_gold_0014_762f7b76cd",
  "schema_version": "rag-chunk-gold-v1",
  "source_schema_version": "rag-doc-gold-v2",
  "split": "train|dev|test",
  "query": "...",
  "question": "...",
  "task_type": "operation|specification|troubleshooting|safety",
  "difficulty": "single-hop",
  "gold_files": ["..."],
  "gold_pages": [4],
  "gold_doc_ids": ["...::p4"],
  "gold_chunk_ids": ["..."],
  "expected_root_ids": ["..."],
  "expected_anchors": ["1.2"],
  "expected_keywords": ["..."],
  "supporting_chunks": [],
  "reference_answer_extract": "...",
  "reference_answer_abstractive": "",
  "quality": {
    "alignment_status": "aligned|failed|manual_review",
    "alignment_confidence": 0.0,
    "quality_score": 0.0,
    "review_status": "draft|reviewed|gold"
  },
  "provenance": {
    "source_dataset": ".jbeval/datasets/rag_doc_gold.jsonl",
    "source_id": "doc_gold_0014_762f7b76cd",
    "index_variant": "title_context",
    "chunker_version": "document_loader_current"
  }
}
```

### Contrastive Training Row

```json
{
  "id": "contrastive_doc_gold_0014_762f7b76cd",
  "schema_version": "rag-contrastive-train-v1",
  "split": "train",
  "query": "...",
  "positive_contexts": [],
  "hard_negatives": [],
  "easy_negatives": [],
  "labels": {
    "positive_relevance": 3,
    "hard_negative_relevance": 0,
    "easy_negative_relevance": 0
  },
  "provenance": {
    "source_chunk_gold_id": "doc_gold_0014_762f7b76cd"
  }
}
```

### Generation Training Row

Generation data should be derived later, after chunk gold is stable:

```json
{
  "id": "generation_doc_gold_0014_762f7b76cd",
  "schema_version": "rag-generation-train-v1",
  "split": "train",
  "query": "...",
  "supporting_chunks": [],
  "reference_answer_extract": "...",
  "reference_answer_abstractive": "...",
  "citation_targets": ["chunk_id"],
  "answer_style": "concise_grounded_zh"
}
```

## Versioning

Use append-only dataset versions. Do not overwrite historical outputs.

Recommended sequence:

- `rag_doc_gold.jsonl`: current canonical seed benchmark.
- `rag_doc_splits_v1.json`: stable source-file split.
- `rag_chunk_gold_v1.jsonl`: first chunk/root aligned version.
- `rag_contrastive_train_v1.jsonl`: first positive/negative training version.
- `rag_generation_train_v1.jsonl`: extractive plus abstractive answer version.
- `rag_multihop_gold_v1.jsonl`: later multi-hop and cross-document extension.

Every generated dataset should have a sibling report:

```text
rag_chunk_gold_v1.report.json
rag_chunk_gold_v1.quality.md
```

Reports should include row count, split count, answer-type distribution, alignment success rate, failed alignment reasons, positive count distribution, negative count distribution, and leakage checks.

## Quality Gates

The first usable training version must pass:

- No source file appears in more than one split.
- Every train/dev/test row has a `split`.
- At least 90% of rows align to one or more chunk ids.
- Every aligned row has at least one `supporting_chunks` item.
- Every contrastive train row has at least one positive and at least two hard negatives.
- No negative chunk id appears in the same row's positive chunk ids.
- No frozen eval row is included in contrastive training output.
- Existing page-level metrics still run against frozen eval.
- New chunk/root metrics run against chunk gold rows.

Rows that fail quality gates should be reported and quarantined, not silently fixed.

## Evaluation Metrics

Keep existing page-level metrics:

- `file_hit@5`
- `page_hit@5`
- `keyword_required_hit@5`
- `anchor_hit@5`
- `mrr`
- `context_precision_id@5`

Add chunk/root-specific metrics:

- `chunk_hit@5`
- `root_hit@5`
- `positive_chunk_mrr`
- `hard_negative_above_positive_rate`
- `positive_context_precision@5`
- `negative_collision_rate`

For generation data, do not use exact-match answer scoring as the only metric. Use extractive answer overlap for smoke checks, but evaluate groundedness and support with citation/chunk evidence.

## Implementation Components

Recommended new scripts:

- `scripts/build_rag_doc_splits.py`
  - Reads `rag_doc_gold.jsonl`.
  - Writes `rag_doc_splits_v1.json`.
  - Balances by source file, answer type, and domain tags.

- `scripts/align_rag_chunk_gold.py`
  - Reads split manifest and current indexed/chunked documents.
  - Writes `rag_chunk_gold_v1.jsonl`.
  - Emits alignment report and quarantined rows.

- `scripts/mine_rag_negatives.py`
  - Reads `rag_chunk_gold_v1.jsonl`.
  - Uses `hard_negative_files` and local retrieval candidates.
  - Writes `rag_contrastive_train_v1.jsonl`.

- `scripts/validate_rag_dataset.py`
  - Validates schema, split leakage, positive/negative collisions, and coverage.
  - Writes `.report.json` and `.quality.md`.

Recommended changes to existing scripts:

- Extend `scripts/evaluate_rag_matrix.py` to prefer `gold_chunk_ids` and `expected_root_ids` when present.
- Keep fallback to `gold_doc_ids`, `expected_anchors`, and `expected_keywords` for page-level rows.

## Phased Plan

### Phase 1: Make the seed set trainable

Deliverables:

- `rag_doc_splits_v1.json`
- `rag_doc_frozen_eval_v1.jsonl`
- `rag_chunk_gold_v1.jsonl`
- `rag_chunk_gold_v1.report.json`
- `rag_chunk_gold_v1.quality.md`

Success criteria:

- File-group split has no leakage.
- At least 90% chunk alignment success.
- Existing retrieval benchmark still runs.
- Chunk/root hit metrics are populated and meaningful.

### Phase 2: Add contrastive training value

Deliverables:

- `rag_contrastive_train_v1.jsonl`
- negative-mining report
- validation report

Success criteria:

- Every train row has positives and negatives.
- Hard negatives are not simple random noise.
- Reranker evaluation can measure whether positives outrank negatives.

### Phase 3: Improve query and answer realism

Deliverables:

- `query_variants` for train rows only.
- `reference_answer_abstractive` for train rows first, then optionally dev.
- Real user query import path if logs are available.

Success criteria:

- No generated variants leak into frozen eval.
- Query variants preserve the same gold chunks.
- Abstractive answers cite or map back to supporting chunks.

### Phase 4: Expand task coverage

Deliverables:

- Multi-page rows.
- Cross-document comparison rows.
- Safety-focused rows.
- More balanced domain coverage.

Success criteria:

- Dataset no longer consists only of single-hop rows.
- Evaluation reports are broken down by task type and difficulty.

## Risks

- Chunk ids can change after chunking/indexing changes. Mitigation: record `index_variant`, chunker version, root ids, anchors, page refs, and source excerpts.
- PDF extraction artifacts may cause fuzzy alignment errors. Mitigation: require page constraints and report ambiguous matches.
- Generated query variants can leak into eval. Mitigation: split first, expand train only.
- Hard negatives can accidentally contain valid evidence. Mitigation: collision checks against source excerpt, gold page, and positive root ids.
- Over-expanding too early will create volume without supervision quality. Mitigation: require chunk gold and negatives before generation expansion.

## Acceptance Checklist

- The current `rag_doc_gold.jsonl` remains unchanged.
- Split manifest exists and is deterministic.
- Frozen eval is source-file isolated.
- Chunk gold rows include chunk/root/supporting evidence.
- Contrastive rows include positives and negatives with no collisions.
- Reports explain failed alignments and quarantined rows.
- Evaluation can separately report page-level and chunk-level performance.

