from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


DEFAULT_RAG_I = "I2"
DEFAULT_RAG_M = "M0"
DEFAULT_RAG_A = "A1"
DEFAULT_RAG_DTYPE = "fp16"


K_HISTORICAL_ALIASES = {
    "K1": "S1_linear",
    "K2": "V3Q",
    "K3": "V3Q_OPT",
}

HISTORICAL_TO_K = {alias.upper(): key for key, alias in K_HISTORICAL_ALIASES.items()}
HISTORICAL_TO_K["S1"] = "K1"

DTYPE_TO_TORCH = {
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
}

DEVICE_ALIASES = {
    "A0": "cpu",
    "A1": "auto",
    "A2": "cuda",
}

VALID_INDEX_PROFILES = {"I1", "I2"}
VALID_MODE_PROFILES = {"M0", "M1", "M2"}


@dataclass(frozen=True)
class RagProfile:
    profile_key: str
    index_key: str
    mode_key: str
    acceleration_key: str
    dtype_key: str
    profile_name: str
    historical_alias: str | None
    rerank_torch_dtype: str
    device_request: str

    def as_metadata(self) -> dict[str, str | None]:
        return {
            "rag_profile": self.profile_name,
            "rag_k": self.profile_key,
            "rag_i": self.index_key,
            "rag_m": self.mode_key,
            "rag_a": self.acceleration_key,
            "rag_dtype": self.dtype_key,
            "legacy_variant": self.historical_alias,
            "profile_key": self.profile_key,
            "index_key": self.index_key,
            "mode_key": self.mode_key,
            "acceleration_key": self.acceleration_key,
            "dtype_key": self.dtype_key,
            "profile_name": self.profile_name,
            "historical_alias": self.historical_alias,
            "rerank_torch_dtype": self.rerank_torch_dtype,
            "device_request": self.device_request,
        }


def _normalize_token(value: str | None, *, default: str) -> str:
    token = (value or default).strip()
    return token.upper()


def canonical_variant_name(value: str | None) -> str:
    token = _normalize_token(value, default="K2")
    if token in K_HISTORICAL_ALIASES:
        return token
    if token in HISTORICAL_TO_K:
        return HISTORICAL_TO_K[token]
    raise ValueError(f"unknown RAG profile variant: {value}")


def resolve_dtype(value: str | None) -> str:
    token = (value or DEFAULT_RAG_DTYPE).strip().lower()
    try:
        return DTYPE_TO_TORCH[token]
    except KeyError as exc:
        raise ValueError(f"unknown RAG dtype: {value}") from exc


def resolve_runtime_dtype(rag_dtype: str | None, rerank_torch_dtype: str | None) -> str:
    return resolve_dtype(rag_dtype or rerank_torch_dtype or DEFAULT_RAG_DTYPE)


def canonical_dtype_key(value: str | None) -> str:
    token = (value or DEFAULT_RAG_DTYPE).strip().lower()
    if token in {"float16", "fp16"}:
        return "fp16"
    if token in {"bfloat16", "bf16"}:
        return "bf16"
    if token in {"float32", "fp32"}:
        return "fp32"
    raise ValueError(f"unknown RAG dtype: {value}")


def resolve_profile(
    rag_k: str | None,
    *,
    rag_i: str | None = None,
    rag_m: str | None = None,
    rag_a: str | None = None,
    dtype: str | None = None,
) -> RagProfile:
    profile_key = canonical_variant_name(rag_k)
    index_key = _normalize_token(rag_i, default=DEFAULT_RAG_I)
    mode_key = _normalize_token(rag_m, default=DEFAULT_RAG_M)
    acceleration_key = _normalize_token(rag_a, default=DEFAULT_RAG_A)
    dtype_key = canonical_dtype_key(dtype)

    if index_key not in VALID_INDEX_PROFILES:
        raise ValueError(f"unknown RAG index profile: {rag_i}")
    if mode_key not in VALID_MODE_PROFILES:
        raise ValueError(f"unknown RAG mode profile: {rag_m}")
    if acceleration_key not in DEVICE_ALIASES:
        raise ValueError(f"unknown RAG acceleration profile: {rag_a}")

    return RagProfile(
        profile_key=profile_key,
        index_key=index_key,
        mode_key=mode_key,
        acceleration_key=acceleration_key,
        dtype_key=dtype_key,
        profile_name=f"{profile_key}/{index_key}/{mode_key}/{acceleration_key}/{dtype_key}",
        historical_alias=K_HISTORICAL_ALIASES.get(profile_key),
        rerank_torch_dtype=resolve_dtype(dtype_key),
        device_request=DEVICE_ALIASES[acceleration_key],
    )


def resolve_variant_profile(variant: str | None, *, dtype: str | None = None) -> RagProfile:
    return resolve_profile(
        variant,
        rag_i=DEFAULT_RAG_I,
        rag_m=DEFAULT_RAG_M,
        rag_a=DEFAULT_RAG_A,
        dtype=dtype,
    )


def profile_config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
