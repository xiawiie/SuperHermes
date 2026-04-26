"""Tests for heading lexical scoring in rag_utils._apply_heading_lexical_scoring."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from query_plan import QueryPlan
from rag_utils import _apply_heading_lexical_scoring


def _make_query_plan(heading_hint="", scope_mode="filter", semantic_query="", anchors=None):
    """Create a minimal QueryPlan for testing."""
    return QueryPlan(
        raw_query=semantic_query or "test",
        semantic_query=semantic_query or heading_hint or "test",
        clean_query=semantic_query or heading_hint or "test",
        doc_hints=[],
        matched_files=[],
        scope_mode=scope_mode,
        heading_hint=heading_hint,
        anchors=anchors or [],
        model_numbers=[],
        intent_type="general",
        route="scoped_hybrid" if scope_mode in {"filter", "boost"} else "global_hybrid",
    )


class TestHeadingLexicalScoring:
    def test_no_heading_hint(self):
        """When heading_hint is empty, candidates are unchanged."""
        plan = _make_query_plan(heading_hint="", scope_mode="filter")
        chunks = [{"chunk_id": "c1", "score": 0.8, "section_path": "安装/配置", "section_title": "安装配置"}]
        result = _apply_heading_lexical_scoring(plan, chunks)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"

    def test_scope_mode_none_skips(self):
        """When scope_mode is none, candidates are unchanged regardless of heading_hint."""
        plan = _make_query_plan(heading_hint="安装配置", scope_mode="none")
        chunks = [{"chunk_id": "c1", "score": 0.8, "section_path": "安装/配置", "section_title": "安装配置"}]
        result = _apply_heading_lexical_scoring(plan, chunks)
        assert result == chunks

    def test_exact_heading_match_ranks_first(self):
        """Exact heading match should boost a chunk to rank first."""
        plan = _make_query_plan(heading_hint="安装配置", scope_mode="filter", semantic_query="安装配置")
        chunks = [
            {"chunk_id": "c1", "section_path": "网络设置", "section_title": "网络设置"},
            {"chunk_id": "c2", "section_path": "", "section_title": "安装配置"},
        ]
        result = _apply_heading_lexical_scoring(plan, chunks)
        assert result[0]["chunk_id"] == "c2"

    def test_partial_heading_match(self):
        """Partial heading match should get smaller boost than exact."""
        plan = _make_query_plan(heading_hint="安装配置", scope_mode="filter", semantic_query="安装配置")
        chunks = [
            {"chunk_id": "c1", "section_path": "", "section_title": "安装配置指南"},
            {"chunk_id": "c2", "section_path": "", "section_title": "完全不相关"},
        ]
        result = _apply_heading_lexical_scoring(plan, chunks)
        assert result[0]["chunk_id"] == "c1"

    def test_section_path_match(self):
        """Section path containing heading_hint should also boost."""
        plan = _make_query_plan(heading_hint="安装", scope_mode="boost", semantic_query="安装")
        chunks = [
            {"chunk_id": "c1", "section_path": "系统安装/配置步骤", "section_title": ""},
            {"chunk_id": "c2", "section_path": "网络设置", "section_title": "网络设置"},
        ]
        result = _apply_heading_lexical_scoring(plan, chunks)
        assert result[0]["chunk_id"] == "c1"

    def test_no_match_no_reorder(self):
        """No match -> single candidate preserved."""
        plan = _make_query_plan(heading_hint="安装配置", scope_mode="filter", semantic_query="安装配置")
        chunks = [
            {"chunk_id": "c1", "section_path": "网络设置", "section_title": "网络设置"},
        ]
        result = _apply_heading_lexical_scoring(plan, chunks)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"

    def test_anchor_boost(self):
        """Anchors matching in heading or section_path should provide extra boost."""
        plan = _make_query_plan(
            heading_hint="配置",
            scope_mode="filter",
            semantic_query="配置",
            anchors=["VLAN"],
        )
        chunks = [
            {"chunk_id": "c1", "section_path": "", "section_title": "VLAN配置"},
            {"chunk_id": "c2", "section_path": "", "section_title": "基本配置"},
        ]
        result = _apply_heading_lexical_scoring(plan, chunks)
        assert result[0]["chunk_id"] == "c1"

    def test_anchor_id_boost(self):
        """Anchors matching anchor_id should also provide extra boost."""
        plan = _make_query_plan(
            heading_hint="配置",
            scope_mode="filter",
            semantic_query="完全不相关",
            anchors=["附录A"],
        )
        chunks = [
            {"chunk_id": "c1", "section_path": "", "section_title": "", "anchor_id": "附录A"},
            {"chunk_id": "c2", "section_path": "", "section_title": "其它内容", "anchor_id": ""},
        ]

        result = _apply_heading_lexical_scoring(plan, chunks)

        assert result[0]["chunk_id"] == "c1"

    def test_empty_candidates(self):
        """Empty candidates list returns empty."""
        plan = _make_query_plan(heading_hint="安装配置", scope_mode="filter")
        result = _apply_heading_lexical_scoring(plan, [])
        assert result == []
