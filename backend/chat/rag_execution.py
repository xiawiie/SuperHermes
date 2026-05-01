from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from backend.rag.runtime_config import load_runtime_config
from backend.rag.trace import RAG_CONTEXT_FORMAT_VERSION
from backend.shared.filename_utils import dedupe_filenames


class RagExecutionPolicy(str, Enum):
    NO_RAG = "no_rag"
    OPTIONAL_TOOL = "optional_tool"
    FORCED_PRELOAD = "forced_preload"


@dataclass(frozen=True)
class RagTurnRequest:
    user_text: str
    context_files: list[str] = field(default_factory=list)
    stream: bool = False


@dataclass(frozen=True)
class RagTurnContext:
    policy: RagExecutionPolicy
    context_files: list[str]
    delivery_mode: str | None = None
    unified_execution_enabled: bool = False
    policy_reason: str = "default"


_DOCUMENT_INTENT_MARKERS = (
    "文档",
    "资料",
    "知识库",
    "检索",
    "根据",
    "引用",
    "source",
    "document",
    "manual",
    "policy",
)


def _looks_like_document_question(user_text: str) -> bool:
    text = (user_text or "").lower()
    return any(marker in text for marker in _DOCUMENT_INTENT_MARKERS)


def plan_rag_turn(
    request: RagTurnRequest,
    *,
    unified_execution_enabled: bool | None = None,
) -> RagTurnContext:
    context_files = dedupe_filenames(request.context_files, max_count=5)
    if unified_execution_enabled is None:
        unified_execution_enabled = load_runtime_config().unified_execution_enabled

    if context_files:
        return RagTurnContext(
            policy=RagExecutionPolicy.FORCED_PRELOAD,
            context_files=context_files,
            delivery_mode="system_message",
            unified_execution_enabled=bool(unified_execution_enabled),
            policy_reason="attached_context_files",
        )

    if unified_execution_enabled and _looks_like_document_question(request.user_text):
        return RagTurnContext(
            policy=RagExecutionPolicy.FORCED_PRELOAD,
            context_files=[],
            delivery_mode="system_message",
            unified_execution_enabled=True,
            policy_reason="document_intent",
        )

    return RagTurnContext(
        policy=RagExecutionPolicy.OPTIONAL_TOOL,
        context_files=[],
        delivery_mode="tool_response",
        unified_execution_enabled=bool(unified_execution_enabled),
        policy_reason="agent_optional_tool",
    )


def mark_rag_execution_policy(
    trace: Mapping[str, Any] | None,
    turn_context: RagTurnContext | None = None,
    *,
    policy: RagExecutionPolicy | str | None = None,
    delivery_mode: str | None = None,
    policy_reason: str | None = None,
) -> dict[str, Any] | None:
    if trace is None:
        return None
    payload = dict(trace)
    resolved_policy = policy or (turn_context.policy if turn_context else None)
    resolved_delivery = delivery_mode or (turn_context.delivery_mode if turn_context else None)
    resolved_reason = policy_reason or (turn_context.policy_reason if turn_context else None)
    if resolved_policy:
        payload["retrieval_policy"] = (
            resolved_policy.value if isinstance(resolved_policy, RagExecutionPolicy) else str(resolved_policy)
        )
    if resolved_delivery:
        payload["context_delivery_mode"] = resolved_delivery
    payload["context_format_version"] = payload.get("context_format_version") or RAG_CONTEXT_FORMAT_VERSION
    if resolved_reason:
        payload["retrieval_policy_reason"] = resolved_reason
    if turn_context is not None:
        payload["rag_unified_execution_enabled"] = turn_context.unified_execution_enabled
    return payload
