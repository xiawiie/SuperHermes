from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from langchain_core.messages import SystemMessage

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


@dataclass(frozen=True)
class RagAnswerResult:
    raw_result: Any
    messages: list[Any]
    execution_mode: str


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


def _with_context_file_instruction(messages: list[Any], context_files: list[str]) -> list[Any]:
    if not context_files:
        return messages
    file_list = "\n".join(f"- {filename}" for filename in context_files)
    instruction = (
        "This turn has attached document context files. "
        "You must call search_knowledge_base before answering, and retrieval is constrained to these filenames:\n"
        f"{file_list}\n"
        "Answer using only the attached files when possible. If they do not contain enough evidence, say so."
    )
    return messages[:-1] + [SystemMessage(content=instruction), messages[-1]]


def _with_retrieved_context_instruction(
    messages: list[Any],
    context_files: list[str],
    retrieved_context: str,
) -> list[Any]:
    file_list = "\n".join(f"- {filename}" for filename in context_files)
    file_section = f"Attached files:\n{file_list}\n\n" if context_files else ""
    if retrieved_context:
        instruction = (
            "You are answering the user's current turn with uploaded document context. "
            "The indexed content below has already been retrieved from the attached files. "
            "Do not say that no document was provided. Do not ask the user to paste the document. "
            "Answer in the user's language using the retrieved context. If the context is insufficient, say exactly what is missing.\n\n"
            f"{file_section}"
            f"Retrieved document context:\n{retrieved_context}"
        )
    elif context_files:
        instruction = (
            "The user attached document files, but no indexed text chunks were retrieved for this turn. "
            "Do not say that no document was provided; say the document was uploaded but no readable indexed text was found, "
            "and suggest checking upload processing or file text extraction.\n\n"
            f"Attached files:\n{file_list}"
        )
    else:
        instruction = (
            "No relevant indexed knowledge-base chunks were retrieved for this turn. "
            "Do not fabricate retrieved content; answer from general knowledge only when appropriate, "
            "or say the indexed knowledge base did not provide enough evidence."
        )
    return messages[:-1] + [SystemMessage(content=instruction), messages[-1]]


def prepare_rag_answer_messages(
    messages: list[Any],
    turn_context: RagTurnContext,
    *,
    retrieved_context: str = "",
) -> list[Any]:
    if turn_context.policy == RagExecutionPolicy.FORCED_PRELOAD:
        return _with_retrieved_context_instruction(
            messages,
            turn_context.context_files,
            retrieved_context,
        )
    return _with_context_file_instruction(messages, turn_context.context_files)


def answer_with_rag_context(
    *,
    messages: list[Any],
    turn_context: RagTurnContext,
    retrieved_context: str,
    agent_instance: Any,
    model_instance: Any,
    recursion_limit: int = 8,
) -> RagAnswerResult:
    prepared_messages = prepare_rag_answer_messages(
        messages,
        turn_context,
        retrieved_context=retrieved_context,
    )
    if turn_context.policy == RagExecutionPolicy.FORCED_PRELOAD:
        raw_result = model_instance.invoke(prepared_messages)
        execution_mode = "preloaded_model"
    else:
        raw_result = agent_instance.invoke(
            {"messages": prepared_messages},
            config={"recursion_limit": recursion_limit},
        )
        execution_mode = "tool_agent"
    return RagAnswerResult(
        raw_result=raw_result,
        messages=prepared_messages,
        execution_mode=execution_mode,
    )


async def stream_answer_with_rag_context(
    *,
    messages: list[Any],
    turn_context: RagTurnContext,
    retrieved_context: str,
    agent_instance: Any,
    model_instance: Any,
    recursion_limit: int = 8,
) -> AsyncIterator[Any]:
    prepared_messages = prepare_rag_answer_messages(
        messages,
        turn_context,
        retrieved_context=retrieved_context,
    )
    if turn_context.policy == RagExecutionPolicy.FORCED_PRELOAD:
        async for msg in model_instance.astream(prepared_messages):
            yield msg
        return

    async for item in agent_instance.astream(
        {"messages": prepared_messages},
        stream_mode="messages",
        config={"recursion_limit": recursion_limit},
    ):
        msg, _metadata = item
        yield msg


def extract_answer_content(result: Any) -> str:
    if isinstance(result, dict):
        if "output" in result:
            return str(result["output"])
        if "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            return str(getattr(msg, "content", str(msg)))
        return str(result)
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


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
