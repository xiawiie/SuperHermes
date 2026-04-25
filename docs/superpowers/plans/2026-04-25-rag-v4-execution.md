# RAG v4 Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete and verify the RAG v4 retrieval and evaluation reform by converging the existing uncommitted implementation.

**Architecture:** Preserve the current `DocumentLoader -> MilvusWriter -> MilvusManager -> retrieve_documents -> rag_pipeline -> evaluate_rag_matrix` boundary. Add controlled query planning, scoped/global hybrid retrieval, metadata-aware rerank pair enrichment, v4 diagnostics, and evaluation metrics without redefining historical baselines or forcing a reindex before gates pass.

**Tech Stack:** Python 3.12, pytest, uv, Milvus/PyMilvus, Redis cache, local BGE-M3 embeddings, local/API reranker, PowerShell commands on Windows.

---

## File Structure

- `backend/query_plan.py`: QueryPlan dataclass, filename normalization, compound filename scoring, registry loading, and parser.
- `backend/rag_utils.py`: single retrieval entry point; semantic query embedding, scoped/global hybrid retrieval, weighted RRF, heading lexical scoring, rerank pair enrichment, rerank cache signatures, and trace fields.
- `backend/rag_pipeline.py`: fallback gate controls, fast-model fallback routing, graph path trace fields.
- `backend/rag_diagnostics.py`: v4 diagnostic categories and confidence classification.
- `backend/document_loader.py`: `title_context_filename` retrieval-text mode for the deferred reindex stage.
- `backend/embedding.py`: BM25 state isolation by `MILVUS_COLLECTION` and `EVAL_RETRIEVAL_TEXT_MODE`.
- `backend/milvus_client.py`: Milvus metadata output fields and unique filename registry query helper.
- `scripts/evaluate_rag_matrix.py`: B0_legacy/S1_linear/S2/S2H/S2HR/S3 variants, v4 metrics, n/a chunk/root reporting, dataset profiles.
- `scripts/analyze_rag_misses.py`: miss classification report and rollups.
- `scripts/derive_natural_query_subset.py`: open-retrieval-natural dataset derivation.
- `docs/rag_evaluation.md`: runbook, gates, metrics, diagnostics, roadmap.
- `.env`: runtime default-off experimental flags for this workspace; `.env.example` remains only a template/reference if already touched by adjacent work.
- `tests/test_*.py`: focused regression coverage for every v4 boundary.

## Task Board

### Task 1: Establish the Current Failure Surface

**Files:**
- Read: `backend/query_plan.py`
- Read: `backend/rag_utils.py`
- Read: `backend/rag_pipeline.py`
- Read: `scripts/evaluate_rag_matrix.py`
- Read: `tests/test_query_plan_parser.py`
- Read: `tests/test_scoped_global_rrf.py`
- Read: `tests/test_rerank_pair_enrichment.py`
- Read: `tests/test_evaluate_rag_matrix.py`

- [ ] **Step 1: Capture targeted test failures**

Run:

```powershell
uv run pytest `
  tests/test_filename_normalization.py `
  tests/test_filename_match_score.py `
  tests/test_query_plan_parser.py `
  tests/test_document_scope_matching.py `
  tests/test_scoped_global_rrf.py `
  tests/test_heading_lexical_scoring.py `
  tests/test_rerank_pair_enrichment.py `
  tests/test_bm25_state_isolation.py `
  tests/test_fallback_disabled_routing.py `
  tests/test_diagnostics_v4.py `
  tests/test_evaluate_rag_matrix.py `
  -q
```

Expected: tests either pass or expose concrete syntax/import/assertion failures to fix in later tasks.

- [ ] **Step 2: Capture broader RAG regression failures**

Run:

```powershell
uv run pytest `
  tests/test_rag_utils.py `
  tests/test_rag_pipeline.py `
  tests/test_rag_pipeline_fast_path.py `
  tests/test_rag_observability.py `
  -q
```

Expected: failures identify integration gaps between existing tests and the v4 implementation.

- [ ] **Step 3: Check Python syntax for edited runtime files**

Run:

```powershell
uv run python -m py_compile `
  backend/query_plan.py `
  backend/rag_utils.py `
  backend/rag_pipeline.py `
  backend/rag_diagnostics.py `
  backend/document_loader.py `
  backend/embedding.py `
  backend/milvus_client.py `
  scripts/evaluate_rag_matrix.py `
  scripts/analyze_rag_misses.py `
  scripts/derive_natural_query_subset.py
```

Expected: no `SyntaxError`; if encoding artifacts cause malformed regex strings, fix them in the relevant task before continuing.

### Task 2: Normalize Variant Names and Evaluation Metrics

**Files:**
- Modify: `scripts/evaluate_rag_matrix.py`
- Modify: `tests/test_evaluate_rag_matrix.py`
- Modify: `docs/rag_evaluation.md`

- [ ] **Step 1: Add or adjust tests for v4 variants**

Ensure `tests/test_evaluate_rag_matrix.py` contains assertions equivalent to:

```python
def test_v4_variant_names_are_available():
    assert "B0_legacy" in VARIANT_CONFIGS
    assert "S1_linear" in VARIANT_CONFIGS
    assert "S2" in VARIANT_CONFIGS
    assert "S2H" in VARIANT_CONFIGS
    assert "S2HR" in VARIANT_CONFIGS
    assert "S3" in VARIANT_CONFIGS

    assert VARIANT_CONFIGS["B0_legacy"]["env"] == {}
    assert VARIANT_CONFIGS["S1_linear"]["env"]["CONFIDENCE_GATE_ENABLED"] == "false"
    assert VARIANT_CONFIGS["S1_linear"]["env"]["RAG_FALLBACK_ENABLED"] == "false"
    assert VARIANT_CONFIGS["S2"]["env"]["DOC_SCOPE_GLOBAL_RESERVE_WEIGHT"] == "0.2"
    assert VARIANT_CONFIGS["S2H"]["env"]["HEADING_LEXICAL_ENABLED"] == "true"
    assert VARIANT_CONFIGS["S2H"]["env"]["HEADING_LEXICAL_WEIGHT"] == "0.20"
    assert VARIANT_CONFIGS["S2HR"]["env"]["RERANK_PAIR_ENRICHMENT_ENABLED"] == "true"
    assert VARIANT_CONFIGS["S3"]["env"]["EVAL_RETRIEVAL_TEXT_MODE"] == "title_context_filename"
    assert VARIANT_CONFIGS["S3"]["requires_reindex"] is True
```

- [ ] **Step 2: Run the variant test and confirm it fails before implementation**

Run:

```powershell
uv run pytest tests/test_evaluate_rag_matrix.py::test_v4_variant_names_are_available -q
```

Expected before fix: fail if `B0_legacy` or `S1_linear` is missing or not configured precisely.

- [ ] **Step 3: Implement v4 variant definitions**

In `scripts/evaluate_rag_matrix.py`, shape the v4 variants like:

```python
"B0_legacy": {
    "description": "current production configuration, no env overrides",
    "reindex_mode": "title_context",
    "requires_reindex": False,
    "env": {},
},
"S1_linear": {
    "description": "gate off + fallback off linear retrieval path",
    "reindex_mode": "title_context",
    "requires_reindex": False,
    "env": {
        "CONFIDENCE_GATE_ENABLED": "false",
        "RAG_FALLBACK_ENABLED": "false",
    },
},
"S2": {
    "description": "QueryPlan + document scope 80/20 parallel hybrid retrieval",
    "reindex_mode": "title_context",
    "requires_reindex": False,
    "env": {
        "CONFIDENCE_GATE_ENABLED": "false",
        "RAG_FALLBACK_ENABLED": "false",
        "DOC_SCOPE_MATCH_FILTER": "0.85",
        "DOC_SCOPE_MATCH_BOOST": "0.60",
        "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.2",
    },
},
```

Keep old variants such as `A0`, `A1`, `B1`, `G0`, `G1`, `G2`, and `G3` intact for backward compatibility.

- [ ] **Step 4: Render Chunk@5 and Root@5 as n/a**

In summary rendering, use a helper equivalent to:

```python
def _fmt_qrels_missing(value: object) -> str:
    return "n/a (qrels missing)"
```

Use it for `Chunk@5` and `Root@5` in the main metrics table, paired comparisons, and diagnostics when chunk/root qrels are absent.

- [ ] **Step 5: Run evaluation tests**

Run:

```powershell
uv run pytest tests/test_evaluate_rag_matrix.py -q
```

Expected: all evaluation-matrix tests pass.

### Task 3: Lock Fallback Default-Off Behavior

**Files:**
- Modify: `backend/rag_utils.py`
- Modify: `backend/rag_pipeline.py`
- Modify: `backend/agent.py`
- Modify: `.env.example`
- Modify: `tests/test_fallback_disabled_routing.py`
- Modify: `tests/test_rag_pipeline.py`
- Modify: `tests/test_rag_pipeline_fast_path.py`

- [ ] **Step 1: Add or verify disabled fallback test**

Ensure `tests/test_fallback_disabled_routing.py` covers:

```python
def test_fallback_disabled_short_circuits_even_when_required(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "RAG_FALLBACK_ENABLED", False)
    state = {
        "rag_trace": {
            "fallback_required": True,
            "retrieved_chunks": [{"chunk_id": "c1"}],
        },
        "context": "context",
        "docs": [{"chunk_id": "c1"}],
    }

    result = rag_pipeline.grade_documents_node(state)
    trace = result["rag_trace"]

    assert result["route"] == "generate_answer"
    assert trace["fallback_required_raw"] is True
    assert trace["fallback_executed"] is False
    assert trace["fallback_disabled"] is True
    assert trace["graph_path"] == "linear_initial_only"
```

- [ ] **Step 2: Implement default-off constants**

In `backend/rag_utils.py`:

```python
CONFIDENCE_GATE_ENABLED = os.getenv("CONFIDENCE_GATE_ENABLED", "false").lower() == "true"
```

In `backend/rag_pipeline.py`:

```python
RAG_FALLBACK_ENABLED = os.getenv("RAG_FALLBACK_ENABLED", "false").lower() == "true"
RAG_FALLBACK_TIMEOUT_SECONDS = float(os.getenv("RAG_FALLBACK_TIMEOUT_SECONDS", "6"))
RAG_FALLBACK_USE_FAST_MODEL = os.getenv("RAG_FALLBACK_USE_FAST_MODEL", "true").lower() != "false"
```

- [ ] **Step 3: Implement grader short-circuit**

In `grade_documents_node`, place the disabled-fallback branch before any LLM grader invocation:

```python
fallback_required_raw = rag_trace.get("fallback_required")
if not RAG_FALLBACK_ENABLED:
    rag_trace.update({
        "grade_score": "skipped_fallback_disabled",
        "grade_route": "generate_answer",
        "rewrite_needed": False,
        "fallback_required_raw": fallback_required_raw,
        "fallback_executed": False,
        "fallback_disabled": bool(fallback_required_raw),
        "graph_path": "linear_initial_only",
    })
    _trace_timings(rag_trace)["grader_ms"] = _elapsed_ms(stage_start)
    return {"route": "generate_answer", "rag_trace": rag_trace}
```

- [ ] **Step 4: Route fallback LLM helpers through FAST_MODEL when enabled**

Implement model selection with the same fallback pattern used by the summary model:

```python
def _get_fallback_llm_model():
    if RAG_FALLBACK_USE_FAST_MODEL:
        model = _get_fast_model()
        if model:
            return model, os.getenv("FAST_MODEL")
    return _get_grader_model(), os.getenv("MODEL")
```

Record `fallback_llm_model` in `rag_trace` for grader/router/stepback/hyde fallback paths.

- [ ] **Step 5: Update `.env.example`**

Add or normalize:

```dotenv
# Experimental RAG fallback features. Default off for production retrieval stability.
CONFIDENCE_GATE_ENABLED=false
RAG_FALLBACK_ENABLED=false
RAG_FALLBACK_TIMEOUT_SECONDS=6
RAG_FALLBACK_USE_FAST_MODEL=true
```

- [ ] **Step 6: Run fallback tests**

Run:

```powershell
uv run pytest `
  tests/test_fallback_disabled_routing.py `
  tests/test_rag_pipeline.py `
  tests/test_rag_pipeline_fast_path.py `
  -q
```

Expected: fallback disabled tests pass and existing fast-path graph behavior remains intact.

### Task 4: Repair QueryPlan Parser and Filename Matching

**Files:**
- Modify: `backend/query_plan.py`
- Modify: `backend/rag_utils.py`
- Modify: `backend/milvus_client.py`
- Modify: `tests/test_query_plan_parser.py`
- Modify: `tests/test_filename_normalization.py`
- Modify: `tests/test_filename_match_score.py`
- Modify: `tests/test_document_scope_matching.py`

- [ ] **Step 1: Ensure parser tests cover final Chinese punctuation and model-number behavior**

Use test cases equivalent to:

```python
def test_semantic_query_removes_model_only_when_file_scope_matches():
    registry = [{"raw": "H3C WX3010H 用户手册.pdf", "normalized": "h3c wx3010h 用户手册"}]
    plan = parse_query_plan("《H3C WX3010H 用户手册》中，如何配置无线？", registry)

    assert plan.scope_mode == "filter"
    assert plan.route == "scoped_hybrid"
    assert "WX3010H" not in plan.semantic_query
    assert "如何配置无线" in plan.semantic_query


def test_semantic_query_keeps_model_when_no_file_scope():
    plan = parse_query_plan("如何配置 WX3010H 无线？", filename_registry=[])

    assert plan.scope_mode == "none"
    assert plan.route == "global_hybrid"
    assert "WX3010H" in plan.semantic_query
```

- [ ] **Step 2: Fix regexes and normalization**

Use valid Unicode regexes in `backend/query_plan.py`:

```python
_BOOK_TITLE_RE = re.compile(r"《(.+?)》")
_MODEL_NUMBER_RE = re.compile(r"[A-Z]{2,}\d{3,}[A-Z0-9]*")
_CHAPTER_RE = re.compile(r"(第\s*\d+\s*章|附录\s*[A-Z\d])")
_BOOK_TITLE_PREFIX_RE = re.compile(r"《[^》]+》\s*(?:中|里|内)?[，,:：]?\s*")
```

Normalize filenames with:

```python
def _normalize_filename(name: str) -> str:
    value = unicodedata.normalize("NFKC", str(name or ""))
    value = os.path.splitext(value)[0]
    value = re.sub(r"\(\d+\)$", "", value)
    value = re.sub(r"（\d+）$", "", value)
    value = re.sub(r"[_\s-]*(副本|copy)$", "", value, flags=re.IGNORECASE)
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value
```

- [ ] **Step 3: Keep weak matches trace-only**

In parser route logic:

```python
if matched_files:
    best_score = matched_files[0][1]
    if best_score >= DOC_SCOPE_MATCH_FILTER:
        scope_mode = "filter"
    elif best_score >= DOC_SCOPE_MATCH_BOOST:
        scope_mode = "boost"
    else:
        scope_mode = "none"

route = "scoped_hybrid" if scope_mode in {"filter", "boost"} else "global_hybrid"
```

- [ ] **Step 4: Preserve explicit context_files as hard boundary**

When `context_files` is provided:

```python
if context_files:
    allowed = {str(name) for name in context_files}
    narrowed = [(filename, score) for filename, score in matched_files if filename in allowed]
    matched_files = narrowed or [(filename, 1.0) for filename in context_files]
    scope_mode = "filter"
```

- [ ] **Step 5: Add Milvus unique filename helper if missing**

In `backend/milvus_client.py`:

```python
def query_unique_filenames(self, filter_expr: str = "") -> list[str]:
    rows = self.query(
        filter_expr=filter_expr,
        output_fields=["filename"],
        limit=QUERY_MAX_LIMIT,
    )
    names = sorted({str(row.get("filename") or "").strip() for row in rows if row.get("filename")})
    return names
```

- [ ] **Step 6: Implement filename registry cache boundaries**

`get_filename_registry()` must use:

- Redis logical key `filename_registry:{MILVUS_COLLECTION}:v{milvus_index_version}` under the existing `superhermes:` prefix, yielding `superhermes:filename_registry:{MILVUS_COLLECTION}:v{milvus_index_version}` in Redis.
- In-process LRU/TTL cache keyed by the same collection/version pair.
- `DOC_SCOPE_FILENAME_REGISTRY_REFRESH_SECONDS=600` default refresh interval.
- `milvus_index_version` invalidation via the existing RedisCache key used by Milvus writes/deletes.

- [ ] **Step 7: Run QueryPlan and filename tests**

Run:

```powershell
uv run pytest `
  tests/test_filename_normalization.py `
  tests/test_filename_match_score.py `
  tests/test_query_plan_parser.py `
  tests/test_document_scope_matching.py `
  -q
```

Expected: all QueryPlan and filename-scope tests pass.

### Task 5: Complete Scoped/Global Hybrid Retrieval

**Files:**
- Modify: `backend/rag_utils.py`
- Modify: `tests/test_scoped_global_rrf.py`
- Modify: `tests/test_rag_utils.py`

- [ ] **Step 1: Ensure weighted RRF tests cover dedupe and weights**

Use tests equivalent to:

```python
def test_weighted_rrf_merge_dedupes_by_chunk_id_and_accumulates_scores():
    scoped = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    global_ = [{"chunk_id": "b"}, {"chunk_id": "c"}]

    result = _weighted_rrf_merge([(scoped, 0.8), (global_, 0.2)], rrf_k=60)

    assert [item["chunk_id"] for item in result] == ["b", "a", "c"]
    assert result[0]["rrf_scope_score"] > result[1]["rrf_scope_score"]
```

- [ ] **Step 2: Embed semantic query once**

In `retrieve_documents`, compute embeddings from `query_plan.semantic_query`:

```python
retrieval_query = query_plan.semantic_query or query
dense_embedding = _embedding_service.get_embeddings([retrieval_query])[0]
sparse_embedding = _embedding_service.get_sparse_embedding(retrieval_query)
```

- [ ] **Step 3: Run scoped/global queries with ThreadPoolExecutor**

Use a helper equivalent to:

```python
with ThreadPoolExecutor(max_workers=2) as pool:
    scoped_future = pool.submit(_scoped_retrieve)
    global_future = pool.submit(_global_retrieve)
    scoped = scoped_future.result()
    global_docs = global_future.result()

retrieved = _weighted_rrf_merge(
    [(scoped, 1.0 - DOC_SCOPE_GLOBAL_RESERVE_WEIGHT), (global_docs, DOC_SCOPE_GLOBAL_RESERVE_WEIGHT)],
    rrf_k=MILVUS_RRF_K,
)
```

If the scoped query raises, use global results and add a stage error:

```python
except Exception as exc:
    stage_errors.append(_stage_error("scoped_hybrid_retrieve", str(exc), "global_hybrid"))
    retrieved = global_docs
```

- [ ] **Step 4: Build filename filter with leaf level**

The filter expression must use:

```python
filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
```

For file filtering, use existing `_build_filename_filter()` or an equivalent quote-safe helper. The final scoped filter must be:

```python
scoped_filter = f"{filter_expr} and {filename_filter}"
```

- [ ] **Step 5: Implement boost mode without hard filtering**

When `query_plan.scope_mode == "boost"`, do not build a Milvus filename filter. Run the global hybrid path using `semantic_query`, then apply `_apply_filename_boost()` before rerank:

```python
if query_plan.scope_mode == "boost":
    retrieved = _apply_filename_boost(query_plan, retrieved)
```

The boost must be traceable with `filename_boost_applied` and `filename_boosted_candidate_count`.

- [ ] **Step 6: Add trace fields**

Set:

```python
rerank_meta.update({
    "query_plan": query_plan.to_dict(),
    "semantic_query": query_plan.semantic_query,
    "scope_mode": query_plan.scope_mode,
    "query_route": query_plan.route,
    "doc_scope_match_ratios": [round(score, 3) for _, score in query_plan.matched_files[:3]],
    "matched_files_top3": [name for name, _ in query_plan.matched_files[:3]],
    "scoped_candidate_count": len(scoped),
    "global_candidate_count": len(global_docs),
    "scope_filter_applied": bool(filename_filter and query_plan.scope_mode == "filter"),
})
```

- [ ] **Step 7: Run scoped retrieval tests**

Run:

```powershell
uv run pytest tests/test_scoped_global_rrf.py tests/test_rag_utils.py -q
```

Expected: scoped/global merge tests pass and existing retrieval behavior remains compatible.

### Task 6: Complete Heading Lexical Scoring and Rerank Pair Enrichment

**Files:**
- Modify: `backend/rag_utils.py`
- Modify: `tests/test_heading_lexical_scoring.py`
- Modify: `tests/test_rerank_pair_enrichment.py`

- [ ] **Step 1: Validate heading scoring tests**

Ensure tests assert that scoring is skipped for `scope_mode="none"` and applied for scoped modes with heading hints.

- [ ] **Step 2: Implement heading lexical score**

Use:

```python
heading_lexical_score = (
    0.5 * SequenceMatcher(None, query_plan.semantic_query, section_path).ratio()
    + 0.3 * SequenceMatcher(None, query_plan.semantic_query, heading).ratio()
    + 0.2 * (1.0 if any(anchor in heading_blob for anchor in query_plan.anchors) else 0.0)
)
final_sort_key = (1 - alpha) * rrf_rank_normalized + alpha * heading_lexical_score
```

Where:

```python
heading_blob = " ".join(str(doc.get(key) or "") for key in ("section_path", "section_title", "anchor_id"))
```

- [ ] **Step 3: Build enriched rerank pair text**

Implement:

```python
def _build_enriched_pair(doc: dict) -> str:
    filename = str(doc.get("filename") or "").strip()
    section_path = str(doc.get("section_path") or doc.get("section_title") or "").strip()
    page = doc.get("page_start", doc.get("page_number", ""))
    anchor = str(doc.get("anchor_id") or "").strip()
    heading = str(doc.get("section_title") or "").strip()
    body = _doc_retrieval_text(doc)
    prefix = f"[{filename}][{section_path}][p.{page}]"
    if anchor:
        prefix += f"[{anchor}]"
    first_line = " ".join(part for part in [prefix, heading] if part).strip()
    return "\n".join(part for part in [first_line, body] if part).strip()
```

- [ ] **Step 4: Use enriched pair in rerank calls**

Before calling local/API/Ollama reranker, build texts with:

```python
pair_texts = [
    _rerank_pair_text(doc, enrichment_enabled=pair_enrichment_enabled)
    for doc in docs_for_rerank
]
```

- [ ] **Step 5: Hash final pair text for rerank cache**

Implement:

```python
def _rerank_doc_signatures(docs: list[dict], enrichment_enabled: bool) -> list[dict]:
    signatures = []
    for doc in docs:
        pair_text = _build_enriched_pair(doc) if enrichment_enabled else _doc_retrieval_text(doc)
        signatures.append({
            "chunk_id": str(doc.get("chunk_id") or doc.get("id") or ""),
            "pair_text_sha1": _sha1_text(pair_text),
        })
    return signatures
```

- [ ] **Step 6: Run ranking quality tests**

Run:

```powershell
uv run pytest tests/test_heading_lexical_scoring.py tests/test_rerank_pair_enrichment.py -q
```

Expected: heading and enrichment tests pass; cache key changes when enrichment output changes.

### Task 7: Complete Diagnostics and Miss Analysis

**Files:**
- Modify: `backend/rag_diagnostics.py`
- Modify: `scripts/analyze_rag_misses.py`
- Modify: `tests/test_diagnostics_v4.py`
- Modify: `docs/rag_evaluation.md`

- [ ] **Step 1: Assert v4 categories**

Ensure tests validate exact categories:

```python
VALID_MISS_CATEGORIES = {
    "file_recall_miss",
    "page_miss",
    "ranking_miss",
    "hard_negative_confusion",
    "low_confidence",
}
```

- [ ] **Step 2: Implement deterministic category order**

Use this order:

```python
if top5 and hard_negative_files and all(doc.get("filename") in hard_negative_files for doc in top5):
    return "hard_negative_confusion"
if not candidate_file_hit:
    return "file_recall_miss"
if candidate_file_hit and not top5_file_hit:
    return "ranking_miss"
if top5_file_hit and expected_pages and not top5_page_hit:
    return "page_miss"
if top1_score < low_confidence_threshold:
    return "low_confidence"
return "none"
```

- [ ] **Step 3: Remove legacy chunk/root miss assumptions from primary output**

Do not emit `chunk_miss` or `dup_root` in primary diagnostics. Preserve raw trace fields only if older reports need compatibility.

- [ ] **Step 4: Add miss analysis rows**

In `scripts/analyze_rag_misses.py`, output:

```python
{
    "category_counts": category_counts,
    "cand_recall_buckets": buckets,
    "fuzzy_histogram": title_filename_ratios,
    "family_confusion": hard_negative_or_family_confusion,
    "anchor_hit_rate": anchor_hit_rate,
    "rerank_drop_top20": rerank_drop_top20,
    "false_retrieval_top10": false_retrieval_top10,
}
```

- [ ] **Step 5: Run diagnostics tests**

Run:

```powershell
uv run pytest tests/test_diagnostics_v4.py tests/test_rag_diagnostics.py -q
```

Expected: v4 diagnostics pass and older diagnostic tests remain compatible where still relevant.

### Task 8: Complete Deferred C3 Prep Without Running Reindex

**Files:**
- Modify: `backend/document_loader.py`
- Modify: `backend/embedding.py`
- Modify: `tests/test_bm25_state_isolation.py`
- Modify: `tests/test_document_loader.py`
- Modify: `.gitignore`
- Modify: `docs/rag_evaluation.md`

- [ ] **Step 1: Isolate BM25 state by collection and text mode**

In `backend/embedding.py`:

```python
_collection = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
_text_mode = os.getenv("EVAL_RETRIEVAL_TEXT_MODE", "title_context")
_DEFAULT_STATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / f"bm25_state_{_collection}_{_text_mode}.json"
)
```

- [ ] **Step 2: Add retrieval text mode for future reindex**

In `backend/document_loader.py`, when mode is `title_context_filename`, produce:

```python
prefix = f"[文档: {short_filename}] [章节: {section_path}] [页: {page_start}]"
if anchor:
    prefix += f" [锚点: {anchor}]"
retrieval_text = f"{prefix}\n{heading}\n{body}".strip()
```

Clamp to 4000 characters with head-plus-tail truncation:

```python
def _limit_retrieval_text(value: str, max_chars: int = 4000) -> str:
    if len(value) <= max_chars:
        return value
    head = value[:3000]
    tail = value[-900:]
    return f"{head}\n...[truncated]...\n{tail}"[:max_chars]
```

- [ ] **Step 3: Ensure gitignore covers BM25 state files**

In `.gitignore`, include:

```gitignore
data/bm25_state*.json
```

- [ ] **Step 4: Run C3 prep tests**

Run:

```powershell
uv run pytest tests/test_bm25_state_isolation.py tests/test_document_loader.py -q
```

Expected: BM25 state isolation and retrieval-text formatting tests pass.

### Task 9: Run Integration Tests and Smoke Evaluation

**Files:**
- Read: `.jbeval/datasets/rag_doc_frozen_eval_v1.jsonl`
- Read: `.jbeval/datasets/rag_doc_gold.jsonl`
- Read: `.jbeval/datasets/rag_doc_gold_natural_v1.jsonl`
- Write by tools: `.jbeval/reports/<run-id>/`

- [ ] **Step 1: Run full targeted unit suite**

Run:

```powershell
uv run pytest `
  tests/test_filename_normalization.py `
  tests/test_filename_match_score.py `
  tests/test_query_plan_parser.py `
  tests/test_document_scope_matching.py `
  tests/test_scoped_global_rrf.py `
  tests/test_heading_lexical_scoring.py `
  tests/test_rerank_pair_enrichment.py `
  tests/test_bm25_state_isolation.py `
  tests/test_fallback_disabled_routing.py `
  tests/test_diagnostics_v4.py `
  tests/test_evaluate_rag_matrix.py `
  tests/test_rag_utils.py `
  tests/test_rag_pipeline.py `
  tests/test_rag_pipeline_fast_path.py `
  tests/test_rag_observability.py `
  -q
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run smoke evaluation**

Run:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile smoke --variants B0_legacy,S1_linear,S2 --skip-reindex --limit 10 --run-id rag-v4-smoke
```

Expected: report files are created under `.jbeval/reports/rag-v4-smoke`; `S1_linear` has `fallback_executed=0`, and its P50 is reported against `B0_legacy` rather than accepted on fallback suppression alone.

- [ ] **Step 3: Run frozen evaluation if services are available**

Run:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile frozen --variants B0_legacy,S1_linear,S2,S2H,S2HR --skip-reindex --run-id rag-v4-frozen
```

Expected: report files are created under `.jbeval/reports/rag-v4-frozen`; metrics can be compared against gates.

- [ ] **Step 4: Run gold and natural evaluation only after smoke/frozen pass**

Run:

```powershell
uv run python scripts\evaluate_rag_matrix.py --dataset-profile gold --variants B0_legacy,S1_linear,S2,S2H,S2HR --skip-reindex --run-id rag-v4-gold
uv run python scripts\evaluate_rag_matrix.py --dataset .jbeval\datasets\rag_doc_gold_natural_v1.jsonl --variants B0_legacy,S1_linear,S2,S2H,S2HR --skip-reindex --run-id rag-v4-natural
```

Expected: gold and natural reports are produced if local Milvus, Redis, embeddings, and reranker are running.

### Task 10: Commit and Report

**Files:**
- Read: `git status --short`
- Read: generated `.jbeval/reports/*/summary.md`
- Modify: `docs/rag_evaluation.md` if evaluation evidence needs recording

- [ ] **Step 1: Review changed files**

Run:

```powershell
git status --short
git diff --stat
```

Expected: changed files are limited to RAG v4 implementation, tests, docs, env example, and generated evaluation artifacts that should be kept.

- [ ] **Step 2: Run whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 3: Stage intentional files only**

Run an explicit `git add -- <file list>` containing only files touched for this task. Do not stage IDE state, `.omx` runtime state, or unrelated frontend changes.

- [ ] **Step 4: Commit with Lore protocol**

Use a commit message shaped like:

```text
Improve RAG retrieval precision without redefining historical baselines

The implementation separates semantic query text from document scope,
keeps fallback disabled by default, and adds no-reindex retrieval and
rerank improvements before any collection migration.

Constraint: Existing worktree contained partially implemented v4 files
Rejected: Reindex first | would make retrieval regressions harder to attribute
Confidence: medium
Scope-risk: moderate
Directive: Do not enable fallback or C3 reindex without comparing S2HR gates
Tested: <unit and eval commands that passed>
Not-tested: <services or datasets unavailable locally>
```

- [ ] **Step 5: Final report**

Report:

- changed files
- simplifications made
- test commands and results
- evaluation report paths and key metrics
- remaining risks
