# RAG Mode Classifier V1 Dataset Quality Upgrade

**Date:** 2026-04-27
**Status:** Approved for planning
**Scope:** Improve `eval/datasets/rag_mode_classifier_v1.jsonl` and its focused evaluator after the first passing version.

## Goal

The first version of `rag_mode_classifier_v1.jsonl` satisfies the baseline gates: at least 100 examples, required label coverage, zero mode mismatches, zero FAST false positives on `must_not_fast=true`, and DEEP trigger rate below 15%.

The second pass should make the dataset harder to overfit and more useful for future classifier implementation. The target is not just a larger dataset; it is a more representative routing evaluation set with stronger hard negatives, less templated language, better DEEP subtype coverage, and more actionable evaluator diagnostics.

## Target Distribution

Prefer a slightly safer DEEP ratio than the maximum allowed by the architecture document.

| Mode | Target |
| --- | ---: |
| FAST | 55-57 |
| STANDARD | 72-76 |
| DEEP | 21-22 |
| Total | 150-153 |
| DEEP trigger rate | Preferably <= 14.3%; hard gate remains < 15% |

The preferred landing point is either:

- 55 FAST / 74 STANDARD / 21 DEEP = 150 total, 14.0% DEEP.
- 56 FAST / 75 STANDARD / 21 DEEP = 152 total, 13.8% DEEP.

Avoid ending with 22 DEEP out of 150 total unless there is a strong reason, because 14.7% leaves little maintenance margin.

## DEEP Subtype Coverage

DEEP must be controlled by both total rate and subtype minimums.

| DEEP subtype | Minimum examples |
| --- | ---: |
| compare / contrast | >= 5 |
| exhaustive / all-items | >= 5 |
| temporal / evolution | >= 4 |
| multi-hop / synthesis | >= 4 |
| explicit deep override | >= 2 |

The subtype labels may remain compatible with the current `complexity_label` scheme, but the evaluator should be able to derive and report this grouped DEEP coverage. For example, `compare` contributes to compare / contrast, `exhaustive_global` and `negation_exhaustive` contribute to exhaustive / all-items where appropriate, and a new `multi_hop_synthesis` label can be added if it improves clarity.

## Routing Semantics To Preserve

User intent modifiers are signals, not all equal overrides.

- User explicit fast, such as "quick answer" or "快速看一下", is a FAST tendency only. It must not override `must_not_fast=true`.
- User explicit fast on complex compare, exhaustive, temporal, negation, or multi-hop queries should route to STANDARD or DEEP according to complexity. In the current scoring design, complex + explicit fast downgrades DEEP to STANDARD rather than forcing FAST.
- User explicit deep, such as "deep mode" or "深入分析", is a stronger override signal. It may route even a simple scoped fact to DEEP when the user explicitly requests depth.
- STANDARD remains the safe default between FAST and DEEP.

These rules should be represented in both examples and evaluator reason checks.

## Sample Improvements

### Add DEEP Hard Examples

Add examples that do not rely only on obvious words like "比较", "所有", or "变化".

Coverage should include:

- Implicit compare: "A 和 B 分别适合什么场景，差别在哪里".
- Cross-document synthesis: combining multiple entities, aspects, or evidence slices.
- Exhaustive global: all-items questions phrased without a fixed `全库里...` prefix.
- Temporal positives: true version/time evolution questions.
- Negation + exhaustive: "除了 X 还有哪些..." and mixed English `except` / `without` forms.

### Add STANDARD Hard Negatives

Split STANDARD hard negatives into two subtypes:

| Type | Purpose |
| --- | --- |
| `scope_ambiguous_standard` | Prevent ambiguous broad questions from being treated as exhaustive DEEP. |
| `single_doc_synthesis_standard` | Prevent single-document summary, issue, and recommendation synthesis from being over-routed to DEEP. |

Examples should include query shapes such as "这个项目有哪些风险？" without all-items intent, and "总结这篇文档的主要问题和建议" within one selected file.

### Add FAST Hard Positives

Add single-file, single-fact, high-scope examples that do not all use "默认" or "是多少". Include:

- Page or section scoped locate/quote questions.
- Parameter lookup with alternate wording.
- Single selected file facts with model numbers that should not become DEEP.

### Rewrite Templated Existing Examples

Replace or rewrite some current template-heavy samples, especially repeated starts like:

- `提取...`
- `总结...`
- `全库里...`

The rewrite should keep labels and expected features correct while making the language more natural.

## Evaluator Enhancements

Keep the evaluator lightweight and independent of Milvus/CrossEncoder. Do not turn this pass into production classifier implementation.

Add diagnostics:

- DEEP subtype minimum coverage report and hard gate.
- Mode proportion report with DEEP `< 15%` as a hard gate and preferred `<= 14.3%` as a warning or soft quality signal.
- Template risk report with top offenders, including:
  - Chinese query prefixes.
  - English query prefixes.
  - query length buckets.
  - label distribution by repeated prefix.
- Changelog-oriented summary fields for added, rewritten, and removed examples if practical.

Retain existing diagnostics:

- total examples and examples by expected mode.
- confusion matrix.
- FAST false-positive rate on `must_not_fast=true`.
- DEEP trigger rate.
- mode reason mismatches.
- top high-risk mismatches.
- feature extraction precision/recall and mismatch examples.

## Changelog Requirement

The implementation report should include a concise change summary similar to:

```text
Added:
- N DEEP compare hard examples
- N DEEP exhaustive hard examples
- N DEEP temporal positive examples
- N DEEP multi-hop/synthesis examples
- N temporal negative examples
- N STANDARD hard negatives
- N FAST hard positives

Rewritten:
- N templated existing queries

Removed:
- N weak or redundant examples
```

This is reviewer-facing evidence that the update improved generalization rather than only satisfying gates.

## Acceptance Gates

Hard gates:

- total examples >= 150.
- required labels all covered.
- each required DEEP subtype meets its minimum threshold.
- DEEP trigger rate < 0.15.
- FAST false-positive rate on `must_not_fast=true` = 0.
- mode mismatch count = 0.
- mode reason mismatch count = 0.
- critical feature mismatch count = 0.
- no FAST predictions for hard-blocked categories.
- focused evaluator exits successfully.
- `tests/test_mode_classifier_dataset.py` passes.
- full `uv run pytest tests/ -q` passes.
- `uv run ruff check backend/ scripts/ tests/` passes.
- `uv run python -m compileall backend scripts` passes.

Quality warnings that should be reported but need not block:

- DEEP trigger rate above 14.3% but below 15%.
- any single template prefix dominating a label category.
- label categories with lower-than-target but still acceptable counts outside the required DEEP subtype gates.

## Out Of Scope

- Implementing the production `backend/rag/modes.py` classifier.
- Enabling active routing.
- Running Milvus or CrossEncoder retrieval evaluation.
- Adding new dependencies.
- Large rewrites of unrelated RAG evaluation infrastructure.

## Implementation Notes

Keep the diff focused:

- Update `eval/datasets/rag_mode_classifier_v1.jsonl`.
- Update `scripts/rag_eval/evaluate_mode_classifier.py` only for dataset-quality diagnostics and gates.
- Update `tests/test_mode_classifier_dataset.py` to validate ratios, subtype minimums, and zero-error gates without overfitting to exact counts unless exact counts are intentionally part of the release evidence.
- Do not modify unrelated dirty files already present in the worktree.
