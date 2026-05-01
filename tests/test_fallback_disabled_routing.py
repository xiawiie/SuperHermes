"""Tests for fallback disabled routing and trace three-part fields."""
from __future__ import annotations




class TestFallbackDisabledRouting:
    def test_fallback_bool_parser_requires_explicit_true(self, monkeypatch):
        """Fallback should only enable when explicitly set to true."""
        from backend.config import env_bool

        monkeypatch.delenv("RAG_FALLBACK_ENABLED", raising=False)
        assert env_bool("RAG_FALLBACK_ENABLED", False) is False

        monkeypatch.setenv("RAG_FALLBACK_ENABLED", "1")
        assert env_bool("RAG_FALLBACK_ENABLED", False) is False

        monkeypatch.setenv("RAG_FALLBACK_ENABLED", "true")
        assert env_bool("RAG_FALLBACK_ENABLED", False) is True

    def test_confidence_gate_bool_parser_requires_explicit_true(self, monkeypatch):
        """Confidence gate should only enable when explicitly set to true."""
        from backend.config import env_bool

        monkeypatch.delenv("CONFIDENCE_GATE_ENABLED", raising=False)
        assert env_bool("CONFIDENCE_GATE_ENABLED", False) is False

        monkeypatch.setenv("CONFIDENCE_GATE_ENABLED", "yes")
        assert env_bool("CONFIDENCE_GATE_ENABLED", False) is False

        monkeypatch.setenv("CONFIDENCE_GATE_ENABLED", "true")
        assert env_bool("CONFIDENCE_GATE_ENABLED", False) is True

    def test_grade_documents_short_circuit_when_disabled(self, monkeypatch):
        """When RAG_FALLBACK_ENABLED=False, grader should short-circuit."""
        from dataclasses import replace

        import backend.rag.pipeline as rag_pipeline
        from backend.rag.runtime_config import load_runtime_config

        monkeypatch.setattr(
            rag_pipeline,
            "load_runtime_config",
            lambda: replace(load_runtime_config({}), fallback_enabled=False),
        )
        state = {
            "question": "test query",
            "context": "test context",
            "rag_trace": {"fallback_required": True},  # Gate thinks fallback needed
        }
        result = rag_pipeline.grade_documents_node(state)
        rag_trace = result.get("rag_trace", {})
        # Even though fallback_required is True, it should be short-circuited.
        assert result.get("route") == "generate_answer"
        assert rag_trace.get("fallback_executed") is False
        assert rag_trace.get("fallback_disabled") is True

    def test_trace_three_part_fields(self, monkeypatch):
        """Verify fallback_required_raw, fallback_executed, fallback_disabled in trace."""
        from dataclasses import replace

        import backend.rag.pipeline as rag_pipeline
        from backend.rag.runtime_config import load_runtime_config

        monkeypatch.setattr(
            rag_pipeline,
            "load_runtime_config",
            lambda: replace(load_runtime_config({}), fallback_enabled=False),
        )
        state = {
            "question": "test query",
            "context": "test context",
            "rag_trace": {"fallback_required": True},
        }
        result = rag_pipeline.grade_documents_node(state)
        rag_trace = result.get("rag_trace", {})
        assert "fallback_required_raw" in rag_trace
        assert "fallback_executed" in rag_trace
        assert "fallback_disabled" in rag_trace
        # fallback_required_raw can be True (gate said yes),
        # but fallback_executed must be False (disabled).
        assert rag_trace["fallback_executed"] is False

    def test_graph_path_trace(self, monkeypatch):
        """When fallback disabled, graph_path should be linear_initial_only."""
        from dataclasses import replace

        import backend.rag.pipeline as rag_pipeline
        from backend.rag.runtime_config import load_runtime_config

        monkeypatch.setattr(
            rag_pipeline,
            "load_runtime_config",
            lambda: replace(load_runtime_config({}), fallback_enabled=False),
        )
        state = {
            "question": "test query",
            "context": "test context",
            "rag_trace": {"fallback_required": True},
        }
        result = rag_pipeline.grade_documents_node(state)
        rag_trace = result.get("rag_trace", {})
        assert rag_trace.get("graph_path") == "linear_initial_only"
