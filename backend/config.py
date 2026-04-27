from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
ARK_API_KEY = os.getenv("ARK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL")
FAST_MODEL = os.getenv("FAST_MODEL")
GRADE_MODEL = os.getenv("GRADE_MODEL", "gpt-4.1")

# --- Evaluation ---
ANSWER_EVAL_GENERATION_MODEL = os.getenv("ANSWER_EVAL_GENERATION_MODEL")
ANSWER_EVAL_JUDGE_MODEL = os.getenv("ANSWER_EVAL_JUDGE_MODEL")

# --- Milvus ---
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "embeddings_collection")

# --- Text Mode ---
EVAL_RETRIEVAL_TEXT_MODE = os.getenv("EVAL_RETRIEVAL_TEXT_MODE", "title_context")

# --- External ---
AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() == "true"


def env_float(name: str, default: float = 0.0) -> float:
    val = os.getenv(name, "").strip()
    return float(val) if val else default


def env_int(name: str, default: int = 0) -> int:
    val = os.getenv(name, "").strip()
    return int(val) if val else default
