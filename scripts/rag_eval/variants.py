from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.rag.profile_naming import resolve_variant_profile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "eval" / "datasets"
DEFAULT_FROZEN_DATASET = DATASET_DIR / "rag_doc_frozen_eval_v1.jsonl"
DEFAULT_GOLD_DATASET = DATASET_DIR / "rag_doc_gold.jsonl"
EVAL_SCHEMA_VERSION = "rag-eval-matrix-v2"
DEFAULT_S3_COLLECTION = "embeddings_collection_v2"
DEFAULT_CANONICAL_CORPUS = PROJECT_ROOT.parent / "doc"
GOLD_TC_COLLECTION = "embeddings_collection_gold_tc"
GOLD_TCF_COLLECTION = "embeddings_collection_gold_tcf"
V3_QUALITY_COLLECTION = "embeddings_collection_v3_quality"
V3_FAST_COLLECTION = "embeddings_collection_v3_fast"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_CE_ENV = {
    "RERANK_MODEL": DEFAULT_RERANK_MODEL,
    "RERANK_PROVIDER": "local",
}
DEFAULT_VARIANTS = "K2,K3"

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "A0": {
        "description": "raw text baseline",
        "reindex_mode": "raw",
        "requires_reindex": True,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "raw",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "A1": {
        "description": "title-context retrieval text",
        "reindex_mode": "title_context",
        "requires_reindex": True,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "B1": {
        "description": "title-context retrieval text with structure rerank",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "G0": {
        "description": "structure rerank without confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
        },
    },
    "G1": {
        "description": "recommended confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.05",
            "LOW_CONF_ROOT_SHARE": "0.45",
            "LOW_CONF_TOP_SCORE": "0.20",
            "ENABLE_ANCHOR_GATE": "true",
        },
    },
    "G2": {
        "description": "looser confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.03",
            "LOW_CONF_ROOT_SHARE": "0.35",
            "LOW_CONF_TOP_SCORE": "0.15",
            "ENABLE_ANCHOR_GATE": "true",
        },
    },
    "G3": {
        "description": "stricter confidence gate",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.08",
            "LOW_CONF_ROOT_SHARE": "0.55",
            "LOW_CONF_TOP_SCORE": "0.25",
            "ENABLE_ANCHOR_GATE": "true",
        },
    },
    "B0_legacy": {
        "description": "current production configuration without evaluation env overrides",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {},
    },
    "B0": {
        "description": "current title-context baseline with configured reranker at final top_k depth",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "0",
        },
    },
    "R1": {
        "description": "configured reranker with deeper rerank top_n=20",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "20",
        },
    },
    "R2": {
        "description": "larger candidate set, deeper rerank, higher Milvus ef",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_CANDIDATE_K": "80",
            "RERANK_TOP_N": "30",
            "MILVUS_SEARCH_EF": "128",
            "MILVUS_RRF_K": "60",
        },
    },
    "P1": {
        "description": "B0 retrieval depth with lower sparse drop ratio",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "0",
            "MILVUS_SPARSE_DROP_RATIO": "0.1",
            "MILVUS_RRF_K": "60",
        },
    },
    "P2": {
        "description": "B0 retrieval depth with larger RRF k",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "0",
            "MILVUS_SPARSE_DROP_RATIO": "0.2",
            "MILVUS_RRF_K": "100",
        },
    },
    "P3": {
        "description": "B0 retrieval depth with lower sparse drop ratio and larger RRF k",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "0",
            "MILVUS_SPARSE_DROP_RATIO": "0.1",
            "MILVUS_RRF_K": "100",
        },
    },
    "F1": {
        "description": "B0 retrieval baseline with confidence gate and fallback enabled",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "false",
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.05",
            "LOW_CONF_ROOT_SHARE": "0.45",
            "LOW_CONF_TOP_SCORE": "0.20",
            "ENABLE_ANCHOR_GATE": "true",
            "RAG_FALLBACK_ENABLED": "true",
            "RERANK_TOP_N": "0",
        },
    },
    "S0": {
        "description": "best retrieval candidate with structure rerank comparison",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RERANK_TOP_N": "20",
        },
    },
    "K1": {
        "description": "K1 stable: gate off + fallback off",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "QUERY_PLAN_ENABLED": "false",
        },
    },
    "S2": {
        "description": "QueryPlan + doc scope 80/20 parallel",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "QUERY_PLAN_ENABLED": "true",
            "DOC_SCOPE_MATCH_FILTER": "0.85",
            "DOC_SCOPE_MATCH_BOOST": "0.60",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.2",
        },
    },
    "S2H": {
        "description": "S2 + heading lexical scoring",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "QUERY_PLAN_ENABLED": "true",
            "DOC_SCOPE_MATCH_FILTER": "0.85",
            "DOC_SCOPE_MATCH_BOOST": "0.60",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.2",
            "HEADING_LEXICAL_ENABLED": "true",
            "HEADING_LEXICAL_WEIGHT": "0.20",
        },
    },
    "S2HR": {
        "description": "S2H + metadata-aware rerank pair enrichment",
        "reindex_mode": "title_context",
        "requires_reindex": False,
        "env": {
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "QUERY_PLAN_ENABLED": "true",
            "DOC_SCOPE_MATCH_FILTER": "0.85",
            "DOC_SCOPE_MATCH_BOOST": "0.60",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.2",
            "HEADING_LEXICAL_ENABLED": "true",
            "HEADING_LEXICAL_WEIGHT": "0.20",
            "RERANK_PAIR_ENRICHMENT_ENABLED": "true",
        },
    },
    "S3": {
        "description": "full: reindex title_context_filename + all enrichments",
        "reindex_mode": "title_context_filename",
        "requires_reindex": True,
        "env": {
            "MILVUS_COLLECTION": DEFAULT_S3_COLLECTION,
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context_filename",
            "STRUCTURE_RERANK_ENABLED": "true",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "QUERY_PLAN_ENABLED": "true",
            "DOC_SCOPE_MATCH_FILTER": "0.85",
            "DOC_SCOPE_MATCH_BOOST": "0.60",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.2",
            "HEADING_LEXICAL_ENABLED": "true",
            "HEADING_LEXICAL_WEIGHT": "0.20",
            "RERANK_PAIR_ENRICHMENT_ENABLED": "true",
        },
    },
    "K2": {
        "description": "K2 quality: isolated full-corpus scoped hybrid + metadata rerank fusion",
        "reindex_mode": "title_context_filename",
        "requires_reindex": True,
        "env": {
            "RAG_INDEX_PROFILE": "v3_quality",
            "MILVUS_COLLECTION": V3_QUALITY_COLLECTION,
            "BM25_STATE_PATH": str(PROJECT_ROOT / "data" / "bm25_state_v3_quality.json"),
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context_filename",
            "STRUCTURE_RERANK_ENABLED": "true",
            "SAME_ROOT_CAP": "3",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "QUERY_PLAN_ENABLED": "true",
            "DOC_SCOPE_MATCH_FILTER": "0.50",
            "DOC_SCOPE_MATCH_BOOST": "0.35",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.15",
            "HEADING_LEXICAL_ENABLED": "true",
            "HEADING_LEXICAL_WEIGHT": "0.15",
            "RERANK_PAIR_ENRICHMENT_ENABLED": "true",
            "RERANK_SCORE_FUSION_ENABLED": "true",
            **DEFAULT_CE_ENV,
            "RERANK_FUSION_RERANK_WEIGHT": "0.65",
            "RERANK_FUSION_RRF_WEIGHT": "0.20",
            "RERANK_FUSION_SCOPE_WEIGHT": "0.10",
            "RERANK_FUSION_METADATA_WEIGHT": "0.05",
            "RAG_CANDIDATE_K": "120",
            "RERANK_TOP_N": "30",
            "RERANK_INPUT_K_GPU": "80",
            "RERANK_INPUT_K_CPU": "30",
            "MILVUS_SEARCH_EF": "160",
            "MILVUS_SPARSE_DROP_RATIO": "0.1",
            "MILVUS_RRF_K": "80",
        },
    },
    "K3": {
        "description": "K3 fast evidence: reduced rerank input + lower EF + smaller candidate set",
        "reindex_mode": "title_context_filename",
        "requires_reindex": False,
        "env": {
            "RAG_INDEX_PROFILE": "v3_quality",
            "MILVUS_COLLECTION": V3_QUALITY_COLLECTION,
            "BM25_STATE_PATH": str(PROJECT_ROOT / "data" / "bm25_state_v3_quality.json"),
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context_filename",
            "STRUCTURE_RERANK_ENABLED": "true",
            "SAME_ROOT_CAP": "3",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "QUERY_PLAN_ENABLED": "true",
            "DOC_SCOPE_MATCH_FILTER": "0.50",
            "DOC_SCOPE_MATCH_BOOST": "0.35",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.15",
            "HEADING_LEXICAL_ENABLED": "true",
            "HEADING_LEXICAL_WEIGHT": "0.15",
            "RERANK_PAIR_ENRICHMENT_ENABLED": "true",
            "RERANK_SCORE_FUSION_ENABLED": "true",
            **DEFAULT_CE_ENV,
            "RERANK_FUSION_RERANK_WEIGHT": "0.65",
            "RERANK_FUSION_RRF_WEIGHT": "0.20",
            "RERANK_FUSION_SCOPE_WEIGHT": "0.10",
            "RERANK_FUSION_METADATA_WEIGHT": "0.05",
            "RAG_CANDIDATE_K": "80",
            "RERANK_TOP_N": "30",
            "RERANK_INPUT_K_GPU": "50",
            "RERANK_INPUT_K_CPU": "30",
            "MILVUS_SEARCH_EF": "128",
            "MILVUS_SPARSE_DROP_RATIO": "0.1",
            "MILVUS_RRF_K": "80",
        },
    },
    "K2_LAYERED": {
        "description": "K2 layered experiment: layered candidate pool with shared rerank/postprocess",
        "reindex_mode": "title_context_filename",
        "requires_reindex": False,
        "env": {
            "RAG_INDEX_PROFILE": "v3_quality",
            "MILVUS_COLLECTION": V3_QUALITY_COLLECTION,
            "BM25_STATE_PATH": str(PROJECT_ROOT / "data" / "bm25_state_v3_quality.json"),
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context_filename",
            "STRUCTURE_RERANK_ENABLED": "true",
            "RERANK_SCORE_FUSION_ENABLED": "true",
            "RERANK_PAIR_ENRICHMENT_ENABLED": "true",
            **DEFAULT_CE_ENV,
            "CONFIDENCE_GATE_ENABLED": "false",
            "QUERY_PLAN_ENABLED": "true",
            "HEADING_LEXICAL_ENABLED": "true",
            "HEADING_LEXICAL_WEIGHT": "0.20",
            "MILVUS_RRF_K": "80",
            "MILVUS_SPARSE_DROP_RATIO": "0.1",
            "MILVUS_SEARCH_EF": "160",
            "RAG_CANDIDATE_K": "80",
            "RERANK_TOP_N": "30",
            "RAG_CANDIDATE_STRATEGY": "layered",
        },
    },
    "V3F": {
        "description": "v3 fast: isolated online-sized scoped hybrid + metadata rerank fusion",
        "reindex_mode": "title_context_filename",
        "requires_reindex": True,
        "env": {
            "RAG_INDEX_PROFILE": "v3_fast",
            "MILVUS_COLLECTION": V3_FAST_COLLECTION,
            "BM25_STATE_PATH": str(PROJECT_ROOT / "data" / "bm25_state_v3_fast.json"),
            "EVAL_RETRIEVAL_TEXT_MODE": "title_context_filename",
            "STRUCTURE_RERANK_ENABLED": "true",
            "SAME_ROOT_CAP": "3",
            "CONFIDENCE_GATE_ENABLED": "false",
            "ENABLE_ANCHOR_GATE": "false",
            "RAG_FALLBACK_ENABLED": "false",
            "QUERY_PLAN_ENABLED": "true",
            "DOC_SCOPE_MATCH_FILTER": "0.50",
            "DOC_SCOPE_MATCH_BOOST": "0.35",
            "DOC_SCOPE_GLOBAL_RESERVE_WEIGHT": "0.15",
            "HEADING_LEXICAL_ENABLED": "true",
            "HEADING_LEXICAL_WEIGHT": "0.15",
            "RERANK_PAIR_ENRICHMENT_ENABLED": "true",
            "RERANK_SCORE_FUSION_ENABLED": "true",
            **DEFAULT_CE_ENV,
            "RERANK_FUSION_RERANK_WEIGHT": "0.65",
            "RERANK_FUSION_RRF_WEIGHT": "0.20",
            "RERANK_FUSION_SCOPE_WEIGHT": "0.10",
            "RERANK_FUSION_METADATA_WEIGHT": "0.05",
            "RAG_CANDIDATE_K": "60",
            "RERANK_TOP_N": "10",
            "RERANK_INPUT_K_GPU": "30",
            "RERANK_INPUT_K_CPU": "10",
            "MILVUS_SEARCH_EF": "96",
            "MILVUS_SPARSE_DROP_RATIO": "0.1",
            "MILVUS_RRF_K": "80",
        },
    },
}


def _profiled_env(base_variant: str, *, profile: str, collection: str, state_name: str) -> dict[str, str]:
    env = {key: str(value) for key, value in VARIANT_CONFIGS[base_variant]["env"].items()}
    env.update(
        {
            "RAG_INDEX_PROFILE": profile,
            "MILVUS_COLLECTION": collection,
            "BM25_STATE_PATH": str(PROJECT_ROOT / "data" / state_name),
        }
    )
    return env


def _profile_fallback_env(base_variant: str, *, candidate_only: bool = False) -> dict[str, str]:
    env = {key: str(value) for key, value in VARIANT_CONFIGS[base_variant]["env"].items()}
    env.update(
        {
            "CONFIDENCE_GATE_ENABLED": "true",
            "LOW_CONF_TOP_MARGIN": "0.05",
            "LOW_CONF_ROOT_SHARE": "0.45",
            "LOW_CONF_TOP_SCORE": "0.20",
            "ENABLE_ANCHOR_GATE": "true",
            "RAG_FALLBACK_ENABLED": "true",
        }
    )
    if candidate_only:
        env["RAG_FALLBACK_CANDIDATE_ONLY"] = "true"
    else:
        env.pop("RAG_FALLBACK_CANDIDATE_ONLY", None)
    return env


def _profile_metadata(
    profile_key: str,
    *,
    legacy_variant: str | None = None,
    display_profile_key: str | None = None,
) -> dict[str, str | None]:
    metadata = resolve_variant_profile(profile_key).as_metadata()
    if display_profile_key is not None:
        metadata["rag_profile"] = metadata["profile_name"] = "{}/{}/{}/{}/{}".format(
            display_profile_key,
            metadata["rag_i"],
            metadata["rag_m"],
            metadata["rag_a"],
            metadata["rag_dtype"],
        )
    if legacy_variant is not None:
        metadata["legacy_variant"] = legacy_variant
        metadata["historical_alias"] = legacy_variant
    return metadata


VARIANT_CONFIGS.update(
    {
        "K1": {
            **VARIANT_CONFIGS["K1"],
            "description": "K1 stable: low-latency structured-index baseline",
            **_profile_metadata("K1"),
        },
        "K2": {
            **VARIANT_CONFIGS["K2"],
            "description": "K2 quality: strong evidence profile with QueryPlan + CE + fusion",
            **_profile_metadata("K2"),
        },
        "K3": {
            **VARIANT_CONFIGS["K3"],
            "description": "K3 fast evidence: reduced candidate and CrossEncoder cost",
            **_profile_metadata("K3"),
        },
        "K2_LAYERED": {
            **VARIANT_CONFIGS["K2_LAYERED"],
            "description": "K2 layered experiment: compatibility target for historical layered variants",
            "experiment_key": "layered",
            **_profile_metadata(
                "K2",
                legacy_variant="V3Q_LAYERED",
                display_profile_key="K2_LAYERED",
            ),
        },
        "GB0": {
            "description": "gold title_context baseline: B0 without v3 routing/fusion",
            "reindex_mode": "title_context",
            "requires_reindex": True,
            "env": _profiled_env(
                "B0",
                profile="gold_tc",
                collection=GOLD_TC_COLLECTION,
                state_name="bm25_state_gold_tc.json",
            ),
        },
        "GS1": {
            "description": "gold title_context baseline: S1 linear path",
            "reindex_mode": "title_context",
            "requires_reindex": False,
            "env": _profiled_env(
                "K1",
                profile="gold_tc",
                collection=GOLD_TC_COLLECTION,
                state_name="bm25_state_gold_tc.json",
            ),
        },
        "GS2": {
            "description": "gold title_context baseline: S2 query plan/doc scope",
            "reindex_mode": "title_context",
            "requires_reindex": False,
            "env": _profiled_env(
                "S2",
                profile="gold_tc",
                collection=GOLD_TC_COLLECTION,
                state_name="bm25_state_gold_tc.json",
            ),
        },
        "GS2H": {
            "description": "gold title_context baseline: S2 + heading lexical",
            "reindex_mode": "title_context",
            "requires_reindex": False,
            "env": _profiled_env(
                "S2H",
                profile="gold_tc",
                collection=GOLD_TC_COLLECTION,
                state_name="bm25_state_gold_tc.json",
            ),
        },
        "GS2HR": {
            "description": "gold title_context baseline: S2H + rerank pair enrichment",
            "reindex_mode": "title_context",
            "requires_reindex": False,
            "env": _profiled_env(
                "S2HR",
                profile="gold_tc",
                collection=GOLD_TC_COLLECTION,
                state_name="bm25_state_gold_tc.json",
            ),
        },
        "GS3": {
            "description": "gold title_context_filename baseline: S3 without v3 routing/fusion",
            "reindex_mode": "title_context_filename",
            "requires_reindex": True,
            "env": _profiled_env(
                "S3",
                profile="gold_tcf",
                collection=GOLD_TCF_COLLECTION,
                state_name="bm25_state_gold_tcf.json",
            ),
        },
    }
)

VARIANT_CONFIGS.update(
    {
        "K2F": {
            **VARIANT_CONFIGS["K2"],
            "description": "K2 quality with confidence gate and full fallback enabled",
            "requires_reindex": False,
            "env": _profile_fallback_env("K2"),
        },
        "K2F_CAND": {
            **VARIANT_CONFIGS["K2"],
            "description": "K2 quality with confidence gate and candidate-only fallback enabled",
            "requires_reindex": False,
            "env": _profile_fallback_env("K2", candidate_only=True),
        },
        "K3F": {
            **VARIANT_CONFIGS["K3"],
            "description": "K3 fast evidence with confidence gate and full fallback enabled",
            "requires_reindex": False,
            "env": _profile_fallback_env("K3"),
        },
        "K3F_CAND": {
            **VARIANT_CONFIGS["K3"],
            "description": "K3 fast evidence with confidence gate and candidate-only fallback enabled",
            "requires_reindex": False,
            "env": _profile_fallback_env("K3", candidate_only=True),
        },
    }
)

LEGACY_VARIANT_ALIASES: dict[str, str] = {
    "S1": "K1",
    "S1_linear": "K1",
    "V3Q": "K2",
    "V3Q_OPT": "K3",
    "V3Q_LAYERED": "K2_LAYERED",
    "EXP_C2": "K2_LAYERED",
    "EXP_C25": "K2_LAYERED",
    "EXP_C3": "K2_LAYERED",
    "EXP_C4": "K2_LAYERED",
    "EXP_C5": "K2_LAYERED",
    "EXP_C7": "K2_LAYERED",
}

for _legacy_variant in LEGACY_VARIANT_ALIASES:
    VARIANT_CONFIGS.pop(_legacy_variant, None)

PAIR_DEFINITIONS = (
    ("A1_vs_A0", "A0", "A1"),
    ("B1_vs_A1", "A1", "B1"),
    ("G1_vs_G0", "G0", "G1"),
    ("G1_vs_G2", "G2", "G1"),
    ("G1_vs_G3", "G3", "G1"),
    ("R1_vs_B0", "B0", "R1"),
    ("R2_vs_R1", "R1", "R2"),
    ("P1_vs_B0", "B0", "P1"),
    ("P2_vs_B0", "B0", "P2"),
    ("P3_vs_B0", "B0", "P3"),
    ("F1_vs_B0", "B0", "F1"),
    ("S0_vs_R1", "R1", "S0"),
    ("K1_vs_B0_legacy", "B0_legacy", "K1"),
    ("S2_vs_K1", "K1", "S2"),
    ("S2H_vs_S2", "S2", "S2H"),
    ("S2HR_vs_S2H", "S2H", "S2HR"),
    ("S3_vs_S2HR", "S2HR", "S3"),
    ("GS1_vs_GB0", "GB0", "GS1"),
    ("GS2_vs_GS1", "GS1", "GS2"),
    ("GS2H_vs_GS2", "GS2", "GS2H"),
    ("GS2HR_vs_GS2H", "GS2H", "GS2HR"),
    ("GS3_vs_GS2HR", "GS2HR", "GS3"),
    ("K2_vs_GS3", "GS3", "K2"),
    ("K2_vs_S3", "S3", "K2"),
    ("V3F_vs_K2", "K2", "V3F"),
    ("K3_vs_K2", "K2", "K3"),
    ("K2F_vs_K2", "K2", "K2F"),
    ("K2F_CAND_vs_K2F", "K2F", "K2F_CAND"),
    ("K3F_vs_K3", "K3", "K3F"),
    ("K3F_CAND_vs_K3F", "K3F", "K3F_CAND"),
)


