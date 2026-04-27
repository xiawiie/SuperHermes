"""Tests for document scope matching: three tiers + context_files priority + route tightening."""
from __future__ import annotations



import backend.rag.query_plan as query_plan
from backend.rag.query_plan import parse_query_plan, DOC_SCOPE_MATCH_FILTER, DOC_SCOPE_MATCH_BOOST


_REGISTRY = [
    {"raw": "H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf", "normalized": "h3c la2608室内无线网关 用户手册-6w100-整本手册"},
    {"raw": "H3C MSR系列开放多业务路由器 Web配置指导(V7)-R6728-6W102-整本手册.pdf", "normalized": "h3c msr系列开放多业务路由器 web配置指导(v7)-r6728-6w102-整本手册"},
    {"raw": "H3C UG系列路由器 用户手册-R0130-6W102-整本手册.pdf", "normalized": "h3c ug系列路由器 用户手册-r0130-6w102-整本手册"},
]


class TestThreeTiers:
    def test_filter_tier_high_score(self):
        """Book title that exactly matches filename -> filter mode."""
        plan = parse_query_plan("《H3C LA2608室内无线网关》中，如何配置？", _REGISTRY)
        assert plan.scope_mode == "filter"
        assert plan.route == "scoped_hybrid"
        if plan.matched_files:
            assert plan.matched_files[0][1] >= DOC_SCOPE_MATCH_FILTER

    def test_none_tier_no_match(self):
        """Query with no matching doc hints -> none mode."""
        plan = parse_query_plan("如何安装操作系统？", _REGISTRY)
        assert plan.scope_mode == "none"
        assert plan.route == "global_hybrid"


class TestContextFilesPriority:
    def test_context_files_override_to_filter(self):
        """User-selected context_files force filter mode regardless of query."""
        plan = parse_query_plan(
            "随便问个问题",
            _REGISTRY,
            context_files=["H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"],
        )
        assert plan.scope_mode == "filter"
        assert plan.route == "scoped_hybrid"
        assert len(plan.matched_files) == 1
        assert plan.matched_files[0][1] == 1.0

    def test_context_files_cannot_escape(self):
        """QueryPlan doc scope only narrows within context_files, cannot escape."""
        plan = parse_query_plan(
            "如何配置",
            _REGISTRY,
            context_files=["H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"],
        )
        # All matched_files should be within context_files
        for f, score in plan.matched_files:
            assert f == "H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"


class TestRouteTightening:
    def test_scope_mode_none_routes_global(self):
        """When scope_mode is none, route must be global_hybrid."""
        plan = parse_query_plan("通用网络问题", _REGISTRY)
        if plan.scope_mode == "none":
            assert plan.route == "global_hybrid"

    def test_scope_mode_filter_routes_scoped(self):
        """When scope_mode is filter, route must be scoped_hybrid."""
        plan = parse_query_plan("《H3C LA2608室内无线网关》中，如何配置？", _REGISTRY)
        if plan.scope_mode == "filter":
            assert plan.route == "scoped_hybrid"

    def test_weak_match_only_in_trace(self):
        """Weak matches (score < 0.60) should only be in trace, not affect routing."""
        registry = [{"raw": "abc-router-guide.pdf", "normalized": "abc router guide"}]
        query_plan.DOC_SCOPE_MATCH_BOOST = 0.80
        try:
            plan = parse_query_plan("《abc manual》怎么配置", registry)
        finally:
            query_plan.DOC_SCOPE_MATCH_BOOST = DOC_SCOPE_MATCH_BOOST

        assert plan.matched_files
        assert plan.matched_files[0][1] < 0.80
        assert plan.scope_mode == "none"
        assert plan.route == "global_hybrid"


class FakeMilvusManager:
    def __init__(self):
        self.calls = 0

    def query_unique_filenames(self, filter_expr):
        self.calls += 1
        self.last_filter_expr = filter_expr
        return ["Alpha Manual.pdf", "Beta Guide.pdf"]


class FakeRegistryCache:
    def __init__(self, version="7"):
        self.version = version
        self.json_values = {}
        self.set_calls = []

    def get_string(self, key):
        if key == "milvus_index_version":
            return self.version
        return None

    def get_json(self, key):
        return self.json_values.get(key)

    def set_json(self, key, value, ttl=None):
        self.set_calls.append((key, value, ttl))
        self.json_values[key] = value


class TestFilenameRegistryCache:
    def setup_method(self):
        query_plan._registry_cache.clear()

    def teardown_method(self):
        query_plan._registry_cache.clear()

    def test_registry_uses_process_cache_and_index_version(self, monkeypatch):
        monkeypatch.setattr("backend.rag.query_plan.MILVUS_COLLECTION", "collection_a")
        fake_cache = FakeRegistryCache(version="7")
        manager = FakeMilvusManager()

        first = query_plan.get_filename_registry(manager, fake_cache)
        second = query_plan.get_filename_registry(manager, fake_cache)

        assert first == second
        assert manager.calls == 1
        assert manager.last_filter_expr == "chunk_level == 3"
        assert fake_cache.set_calls[0][0] == "filename_registry:collection_a:v7"

        fake_cache.version = "8"
        query_plan.get_filename_registry(manager, fake_cache)
        assert manager.calls == 2

    def test_registry_reads_redis_cache_before_milvus(self, monkeypatch):
        monkeypatch.setattr("backend.rag.query_plan.MILVUS_COLLECTION", "collection_b")
        fake_cache = FakeRegistryCache(version="3")
        fake_cache.json_values["filename_registry:collection_b:v3"] = [
            {"raw": "Cached.pdf", "normalized": "cached"},
        ]
        manager = FakeMilvusManager()

        entries = query_plan.get_filename_registry(manager, fake_cache)

        assert entries == [{"raw": "Cached.pdf", "normalized": "cached"}]
        assert manager.calls == 0
