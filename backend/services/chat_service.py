"""Service-level compatibility wrappers for chat execution."""

from backend.chat.agent import chat_with_agent, chat_with_agent_stream


def run_chat(message: str, username: str, session_id: str, context_files: list[str] | None = None):
    return chat_with_agent(message, username, session_id, context_files or [])


def run_chat_stream(
    message: str,
    username: str,
    session_id: str,
    regenerate: bool = False,
    context_files: list[str] | None = None,
):
    return chat_with_agent_stream(message, username, session_id, regenerate, context_files or [])
