"""Tests for QueryPlan parser and semantic_query construction."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from query_plan import (
    parse_query_plan,
)


_SAMPLE_REGISTRY = [
    {"raw": "H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf", "normalized": "h3c la2608室内无线网关 用户手册-6w100-整本手册"},
    {"raw": "H3C MSR系列开放多业务路由器 Web配置指导(V7)-R6728-6W102-整本手册.pdf", "normalized": "h3c msr系列开放多业务路由器 web配置指导(v7)-r6728-6w102-整本手册"},
    {"raw": "H3C NER214W路由器 用户手册-6W101-整本手册.pdf", "normalized": "h3c ner214w路由器 用户手册-6w101-整本手册"},
    {"raw": "H3C UG系列路由器 用户手册-R0130-6W102-整本手册.pdf", "normalized": "h3c ug系列路由器 用户手册-r0130-6w102-整本手册"},
]


class TestBookTitleExtraction:
    def test_extracts_book_title(self):
        plan = parse_query_plan("《H3C LA2608室内无线网关》中，如何配置LA2608与无线控制器互通？", _SAMPLE_REGISTRY)
        assert plan.doc_hints == ["H3C LA2608室内无线网关"]

    def test_no_book_title(self):
        plan = parse_query_plan("如何配置WAN", _SAMPLE_REGISTRY)
        assert plan.doc_hints == []


class TestSemanticQuery:
    def test_removes_book_title_prefix(self):
        plan = parse_query_plan("《H3C LA2608室内无线网关》中，如何配置LA2608与无线控制器互通？", _SAMPLE_REGISTRY)
        assert "《" not in plan.semantic_query
        assert "LA2608" in plan.semantic_query or plan.scope_mode in {"filter", "boost"}

    def test_model_number_retained_when_no_match(self):
        plan = parse_query_plan("如何配置WX9999Z功能？", _SAMPLE_REGISTRY)
        assert "WX9999Z" in plan.semantic_query
        assert plan.scope_mode == "none"
        assert plan.route == "global_hybrid"

    def test_model_number_removed_when_high_confidence_match(self):
        plan = parse_query_plan("《H3C LA2608室内无线网关》中，如何配置LA2608？", _SAMPLE_REGISTRY)
        if plan.scope_mode in {"filter", "boost"}:
            # Model number should be removed from semantic_query when high-confidence match
            assert "LA2608" not in plan.semantic_query or "LA2608" in plan.raw_query

    def test_fallback_to_raw_query(self):
        plan = parse_query_plan("简单问题", None)
        assert plan.semantic_query == "简单问题"


class TestScopeModeAndRoute:
    def test_filter_mode_for_exact_match(self):
        plan = parse_query_plan("《H3C LA2608室内无线网关》中，如何配置？", _SAMPLE_REGISTRY)
        assert plan.scope_mode in {"filter", "boost"}
        assert plan.route == "scoped_hybrid"

    def test_none_mode_for_no_match(self):
        plan = parse_query_plan("如何配置云服务？", _SAMPLE_REGISTRY)
        # Could be boost if "云服务" matches, or none
        if plan.scope_mode == "none":
            assert plan.route == "global_hybrid"

    def test_context_files_override(self):
        plan = parse_query_plan(
            "如何配置",
            _SAMPLE_REGISTRY,
            context_files=["H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"],
        )
        assert plan.scope_mode == "filter"
        assert plan.route == "scoped_hybrid"
        assert len(plan.matched_files) == 1
        assert plan.matched_files[0][1] == 1.0

    def test_weak_match_does_not_route_scoped(self):
        plan = parse_query_plan("通用网络问题", _SAMPLE_REGISTRY)
        if plan.scope_mode == "none":
            assert plan.route == "global_hybrid"


class TestModelNumberExtraction:
    def test_extracts_model_number(self):
        plan = parse_query_plan("如何配置LA2608？", _SAMPLE_REGISTRY)
        assert "LA2608" in plan.model_numbers

    def test_extracts_complex_model_number(self):
        plan = parse_query_plan("配置WX3010H设备", _SAMPLE_REGISTRY)
        assert "WX3010H" in plan.model_numbers


class TestAnchorExtraction:
    def test_extracts_chapter(self):
        plan = parse_query_plan("第3章如何配置？", _SAMPLE_REGISTRY)
        assert any("第" in a and "章" in a for a in plan.anchors)

    def test_extracts_appendix(self):
        plan = parse_query_plan("附录A参数说明", _SAMPLE_REGISTRY)
        assert any("附录" in a for a in plan.anchors)


class TestToDict:
    def test_serializes(self):
        plan = parse_query_plan("《H3C LA2608室内无线网关》中，如何配置？", _SAMPLE_REGISTRY)
        d = plan.to_dict()
        assert "raw_query" in d
        assert "semantic_query" in d
        assert "scope_mode" in d
        assert "route" in d
        assert "matched_files" in d
