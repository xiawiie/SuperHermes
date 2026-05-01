from __future__ import annotations

from typing import Any

from scripts.rag_eval.metrics import QRELS_NA


def fmt_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value if value == QRELS_NA else str(value)[:12]
    if isinstance(value, bool):
        return "1.000" if value else "0.000"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _variant_label(summary: dict[str, Any], variant: str) -> str:
    fingerprint = summary.get("build_fingerprint") or {}
    variants = fingerprint.get("variants") or {}
    if not isinstance(variants, dict):
        variants = {}
    metadata = variants.get(variant) or {}
    label = metadata.get("profile_name") or variant
    alias = metadata.get("historical_alias")
    if alias and alias != variant:
        return f"{label} (historical: {alias})"
    return str(label)


def _fmt_distribution(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "-"
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RAG Matrix Evaluation Summary",
        "",
        f"Generated at: `{summary.get('generated_at')}`",
        f"Rows: `{summary.get('sample_rows')}`",
        "",
    ]
    fingerprint = summary.get("build_fingerprint") or {}
    if fingerprint:
        lines.extend(
            [
                "## Build Info",
                "",
                f"- Git commit: `{fingerprint.get('git_commit') or 'unknown'}`",
                f"- Corpus path: `{fingerprint.get('corpus_path')}`",
                f"- Dataset SHA256: `{fingerprint.get('dataset_sha256')}`",
                f"- Index build timestamp: `{fingerprint.get('index_build_timestamp') or 'not rebuilt in this run'}`",
                "",
            ]
        )
        variant_fingerprints = fingerprint.get("variants") or {}
        if isinstance(variant_fingerprints, dict) and variant_fingerprints:
            lines.extend(["### RAG Profiles", ""])
            for variant, metadata in variant_fingerprints.items():
                alias = metadata.get("historical_alias") or "-"
                lines.append(
                    "- `{variant}` => `{profile}`; historical=`{alias}`; collection=`{collection}`; hash=`{hash}`".format(
                        variant=variant,
                        profile=metadata.get("profile_name") or variant,
                        alias=alias,
                        collection=metadata.get("collection"),
                        hash=metadata.get("profile_config_hash"),
                    )
                )
            lines.append("")
    coverage_reports = summary.get("coverage_preflight") or []
    if coverage_reports:
        lines.extend(["## Coverage Preflight", ""])
        for report in coverage_reports:
            report_type = report.get("type")
            if report_type == "corpus_file_coverage":
                lines.append(
                    "- Corpus file coverage: {covered}/{expected} ({coverage}) at `{path}`".format(
                        covered=report.get("covered_files"),
                        expected=report.get("expected_unique_files"),
                        coverage=fmt_metric(report.get("coverage")),
                        path=report.get("documents_dir"),
                    )
                )
            elif report_type == "qrel_coverage":
                lines.append(
                    "- Qrel coverage: file={file}, page={page}, chunk={chunk} ({chunk_rows}/{rows}), root={root} ({root_rows}/{rows})".format(
                        file=fmt_metric(report.get("file_qrel_coverage")),
                        page=fmt_metric(report.get("page_qrel_coverage")),
                        chunk=fmt_metric(report.get("chunk_qrel_coverage")),
                        chunk_rows=report.get("chunk_qrel_rows", report.get("chunk_qrel_samples", 0)),
                        root=fmt_metric(report.get("root_qrel_coverage")),
                        root_rows=report.get("root_qrel_rows", report.get("root_qrel_samples", 0)),
                        rows=report.get("total_rows", report.get("rows", 0)),
                    )
                )
            elif report_type == "external_chunk_qrels":
                lines.append(
                    "- External qrels: `{source}` matched={matched}/{rows}, chunk_mode={chunk_mode}, root_mode={root_mode}, "
                    "conflicts={conflicts}, review={review}".format(
                        source=report.get("qrel_source") or "none",
                        matched=report.get("matched_rows", 0),
                        rows=report.get("dataset_rows", 0),
                        chunk_mode=report.get("chunk_qrel_match_mode"),
                        root_mode=report.get("root_qrel_match_mode"),
                        conflicts=report.get("conflict_count", 0),
                        review=fmt_metric(report.get("review_coverage")),
                    )
                )
            elif report_type == "collection_closure":
                metadata = report.get("metadata_coverage") or {}
                lines.append(
                    "- `{variant}` collection `{collection}` coverage: {covered}/{expected} ({coverage}); "
                    "page_meta={page_meta}, retrieval_text={retrieval_text}, p95_len={p95}".format(
                        variant=report.get("variant"),
                        collection=report.get("collection"),
                        covered=report.get("covered_files"),
                        expected=report.get("expected_unique_files"),
                        coverage=fmt_metric(report.get("coverage")),
                        page_meta=fmt_metric(metadata.get("page_number_or_page_start_rate")),
                        retrieval_text=fmt_metric(metadata.get("retrieval_text_non_empty_rate")),
                        p95=fmt_metric(metadata.get("retrieval_text_length_p95")),
                    )
                )
        lines.append("")
    lines.extend([
        "## Variant Metrics",
        "",
        "| Variant | Rows | File@5 | File+Page@5 | FileCandRecall | HardNeg@5 | Anchor@5 | MRR | Chunk@5 | Root@5 | P50 ms | P95 ms | Error | FallbackReq | FallbackExec | Helped | Hurt |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for variant, metrics in summary.get("variants", {}).items():
        lines.append(
            "| {variant} | {rows} | {file} | {file_page} | {candidate_recall} | {hard_neg} | {anchor} | {mrr} | {chunk} | {root} | {p50} | {p95} | {error} | {fallback} | {fallback_executed} | {fallback_helped} | {fallback_hurt} |".format(
                variant=_variant_label(summary, variant),
                rows=metrics.get("rows", 0),
                file=fmt_metric(metrics.get("file_hit_at_5")),
                file_page=fmt_metric(metrics.get("file_page_hit_at_5")),
                candidate_recall=fmt_metric(metrics.get("file_candidate_recall_before_rerank")),
                hard_neg=fmt_metric(metrics.get("hard_negative_context_ratio_at_5")),
                anchor=fmt_metric(metrics.get("anchor_hit_at_5")),
                mrr=fmt_metric(metrics.get("mrr")),
                chunk=fmt_metric(metrics.get("chunk_hit_at_5")),
                root=fmt_metric(metrics.get("root_hit_at_5")),
                p50=fmt_metric(metrics.get("p50_latency_ms")),
                p95=fmt_metric(metrics.get("p95_latency_ms")),
                error=fmt_metric(metrics.get("error_rate")),
                fallback=fmt_metric(metrics.get("fallback_trigger_rate")),
                fallback_executed=fmt_metric(metrics.get("fallback_executed_rate")),
                fallback_helped=fmt_metric(metrics.get("fallback_helped_rate")),
                fallback_hurt=fmt_metric(metrics.get("fallback_hurt_rate")),
            )
        )

    if any(
        metrics.get("rerank_enabled_rate") is not None
        or metrics.get("rerank_applied_rate") is not None
        or metrics.get("ce_predict_executed_rate") is not None
        for metrics in summary.get("variants", {}).values()
    ):
        lines.extend([
            "",
            "## Rerank / CE",
            "",
            "| Variant | RerankEnabled | RerankApplied | CEPredictExec | CECacheHit | CEInputAvg | CEP50 ms | CEP95 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for variant, metrics in summary.get("variants", {}).items():
            lines.append(
                "| {variant} | {enabled} | {applied} | {executed} | {cache_hit} | {input_avg} | {p50} | {p95} |".format(
                    variant=_variant_label(summary, variant),
                    enabled=fmt_metric(metrics.get("rerank_enabled_rate")),
                    applied=fmt_metric(metrics.get("rerank_applied_rate")),
                    executed=fmt_metric(metrics.get("ce_predict_executed_rate")),
                    cache_hit=fmt_metric(metrics.get("ce_cache_hit_rate")),
                    input_avg=fmt_metric(metrics.get("avg_ce_input_count")),
                    p50=fmt_metric(metrics.get("p50_ce_latency_ms")),
                    p95=fmt_metric(metrics.get("p95_ce_latency_ms")),
                )
            )

    if any(
        metrics.get("candidate_strategy_requested_distribution")
        or metrics.get("candidate_strategy_effective_distribution")
        or metrics.get("rerank_execution_mode_distribution")
        for metrics in summary.get("variants", {}).values()
    ):
        lines.extend([
            "",
            "## Candidate Strategy",
            "",
            "| Variant | Requested | Effective | RerankExecution |",
            "| --- | --- | --- | --- |",
        ])
        for variant, metrics in summary.get("variants", {}).items():
            lines.append(
                "| {variant} | {requested} | {effective} | {rerank_mode} |".format(
                    variant=_variant_label(summary, variant),
                    requested=_fmt_distribution(metrics.get("candidate_strategy_requested_distribution")),
                    effective=_fmt_distribution(metrics.get("candidate_strategy_effective_distribution")),
                    rerank_mode=_fmt_distribution(metrics.get("rerank_execution_mode_distribution")),
                )
            )

    lines.extend([
        "",
        "## Qrel Coverage",
        "",
        "| Variant | FileQrel | PageQrel | ChunkQrel | RootQrel |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for variant, metrics in summary.get("variants", {}).items():
        lines.append(
            "| {variant} | {file_qrel} | {page_qrel} | {chunk_qrel} | {root_qrel} |".format(
                variant=_variant_label(summary, variant),
                file_qrel=fmt_metric(metrics.get("file_qrel_coverage")),
                page_qrel=fmt_metric(metrics.get("page_qrel_coverage")),
                chunk_qrel=fmt_metric(metrics.get("chunk_qrel_coverage")),
                root_qrel=fmt_metric(metrics.get("root_qrel_coverage")),
            )
        )

    lines.extend([
        "",
        "## Experimental Evidence Metrics",
        "",
        "| Variant | ChunkRows | RootRows | ChunkMRR | RootMRR | AnswerSupport@5 exp |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for variant, metrics in summary.get("variants", {}).items():
        lines.append(
            "| {variant} | {chunk_rows} | {root_rows} | {chunk_mrr} | {root_mrr} | {answer_support} |".format(
                variant=_variant_label(summary, variant),
                chunk_rows=metrics.get("chunk_qrel_rows", 0),
                root_rows=metrics.get("root_qrel_rows", 0),
                chunk_mrr=fmt_metric(metrics.get("chunk_mrr")),
                root_mrr=fmt_metric(metrics.get("root_mrr")),
                answer_support=fmt_metric(metrics.get("answer_support_hit_at_5_experimental")),
            )
        )

    lines.extend(["", "## Paired Comparisons", ""])
    for pair_name, counts in summary.get("paired_comparisons", {}).items():
        lines.append(
            f"- `{pair_name}`: wins={counts.get('wins', 0)}, losses={counts.get('losses', 0)}, ties={counts.get('ties', 0)}, missing={counts.get('missing', 0)}"
        )

    lines.extend(["", "## Diagnostics", ""])
    for variant, counts in summary.get("diagnostics", {}).items():
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        lines.append(f"- `{variant}`: {rendered or 'none'}")

    lines.extend(["", "## Rewrite Strategies", ""])
    for variant, metrics in summary.get("variants", {}).items():
        counts = metrics.get("rewrite_strategy_distribution") or {}
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        lines.append(f"- `{variant}`: {rendered or 'none'}")

    if any(metrics.get("faithfulness_score") is not None for metrics in summary.get("variants", {}).values()):
        lines.extend([
            "",
            "## Answer Evaluation",
            "",
            "| Variant | Faithfulness | AnswerRelevance | CitationCoverage | AnswerEvalError |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for variant, metrics in summary.get("variants", {}).items():
            lines.append(
                "| {variant} | {faithfulness} | {answer_relevance} | {citation_coverage} | {answer_error} |".format(
                    variant=_variant_label(summary, variant),
                    faithfulness=fmt_metric(metrics.get("faithfulness_score")),
                    answer_relevance=fmt_metric(metrics.get("answer_relevance_score")),
                    citation_coverage=fmt_metric(metrics.get("citation_coverage")),
                    answer_error=fmt_metric(metrics.get("answer_eval_error_rate")),
                )
            )

    lines.extend([
        "",
        "## Notes",
        "",
        "- `FileCandRecall` is strict filename-level recall before rerank; `weak_any_candidate_recall_before_rerank` remains in JSON for legacy diagnostics.",
        "- Primary metrics: `File@5`, `File+Page@5`, `FileCandRecall`, `Chunk@5`, `Root@5`, `HardNeg@5`, `Anchor@5`, `MRR`, `P50/P95`.",
        "- `HardNeg@5` is only meaningful on the gold dataset (hard_negative_files defined); shown as `-` on natural/frozen.",
        "- DeepEval/Ragas/TruLens are not runtime dependencies in this report.",
    ])
    return "\n".join(lines) + "\n"
