from dotenv import load_dotenv
import os
import json
import asyncio
import threading
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage
from conversation_storage import ConversationStorage
from tools import get_current_weather, search_knowledge_base, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue

load_dotenv()

API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
_agent_lock = threading.Lock()
agent = None
model = None

def create_agent_instance():
    from langchain.agents import create_agent
    from langchain.chat_models import init_chat_model

    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True,
    )

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base],
        system_prompt=(
            "You are a Super Cute Pony Bot that loves to help users. "
            "When responding, you may use tools to assist. "
            "Use search_knowledge_base when users ask document/knowledge questions. "
            "Do not call the same tool repeatedly in one turn. At most one knowledge tool call per turn. "
            "Once you call search_knowledge_base and receive its result, you MUST immediately produce the Final Answer based on that result. "
            "After receiving search_knowledge_base result, you MUST NOT call any tool again (including get_current_weather or search_knowledge_base). "
            "If the retrieved context is insufficient, answer honestly that you don't know instead of making up facts. "
            "If tool results include a Step-back Question/Answer, use that general principle to reason and answer, "
            "but do not reveal chain-of-thought. "
            "If you don't know the answer, admit it honestly."
        ),
    )
    return agent, model


def get_agent_instance():
    global agent, model
    if agent is None or model is None:
        with _agent_lock:
            if agent is None or model is None:
                agent, model = create_agent_instance()
    return agent, model

storage = ConversationStorage()

def summarize_old_messages(model, messages: list) -> str:
    """将旧消息总结为摘要"""
    # 提取旧对话
    old_conversation = "\n".join([
        f"{'用户' if msg.type == 'human' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    # 生成摘要
    summary_prompt = f"""请总结以下对话的关键信息：

{old_conversation}
总结（包含用户信息、重要事实、待办事项）："""

    summary = model.invoke(summary_prompt).content
    return summary


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并返回响应"""
    agent_instance, model_instance = get_agent_instance()
    messages = storage.load(user_id, session_id)
    compacted_history = False

    # 清理可能残留的 RAG 上下文，避免跨请求污染
    get_last_rag_context(clear=True)
    reset_tool_call_guards()
    
    if len(messages) > 50:
        summary = summarize_old_messages(model_instance, messages[:40])

        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]
        compacted_history = True

    user_message = HumanMessage(content=user_text)
    messages.append(user_message)
    result = agent_instance.invoke(
        {"messages": messages},
        config={"recursion_limit": 8},
    )

    response_content = ""
    if isinstance(result, dict):
        if "output" in result:
            response_content = result["output"]
        elif "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            response_content = getattr(msg, "content", str(msg))
        else:
            response_content = str(result)
    elif hasattr(result, "content"):
        response_content = result.content
    else:
        response_content = str(result)
    
    ai_message = AIMessage(content=response_content)
    messages.append(ai_message)

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    if compacted_history:
        extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
        storage.save(user_id, session_id, messages, extra_message_data=extra_message_data)
    else:
        storage.append_messages(
            user_id,
            session_id,
            [user_message, ai_message],
            extra_message_data=[None, {"rag_trace": rag_trace}],
        )

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


async def chat_with_agent_stream(
    user_text: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    regenerate: bool = False,
):
    """使用 Agent 处理用户消息并流式返回响应。
    
    架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送，
    而非等待工具完成后才显示。
    """
    agent_instance, model_instance = get_agent_instance()
    messages = await asyncio.to_thread(storage.load, user_id, session_id)
    compacted_history = False

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    if len(messages) > 50:
        summary = await asyncio.to_thread(summarize_old_messages, model_instance, messages[:40])
        messages = [
            SystemMessage(content=f"之前的对话摘要：\n{summary}")
        ] + messages[40:]
        compacted_history = True

    regenerate = bool(regenerate)
    user_message = None
    if regenerate:
        if not messages or getattr(messages[-1], "type", None) != "ai":
            raise ValueError("无法重新生成：上一条不是助手回复")
        messages.pop()
        if not messages or getattr(messages[-1], "type", None) != "human":
            raise ValueError("无法重新生成：缺少对应的用户消息")
    else:
        user_message = HumanMessage(content=user_text)
        messages.append(user_message)

    full_response = ""

    async def _agent_worker():
        """后台任务：运行 agent 并将内容 chunk 推入输出队列。"""
        nonlocal full_response
        try:
            async for msg, metadata in agent_instance.astream(
                {"messages": messages},
                stream_mode="messages",
                config={"recursion_limit": 8},
            ):
                if not isinstance(msg, AIMessageChunk):
                    continue
                if getattr(msg, "tool_call_chunks", None):
                    continue

                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, str):
                            content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")

                if content:
                    full_response += content
                    await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        # RAG 步骤在工具执行期间通过 call_soon_threadsafe 实时入队，不需要等 agent 产出 chunk
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass  # 任务已成功取消
        raise  # 重新抛出 GeneratorExit 以便 FastAPI 正确处理关闭
    finally:
        # 正常结束或异常退出时清理
        set_rag_step_queue(None)
        if not agent_task.done():
             agent_task.cancel()

    # 获取 RAG trace
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"

    # 保存对话
    ai_message = AIMessage(content=full_response)
    messages.append(ai_message)
    if compacted_history:
        extra_message_data = [None] * (len(messages) - 1) + [{"rag_trace": rag_trace}]
        await asyncio.to_thread(
            storage.save,
            user_id,
            session_id,
            messages,
            None,
            extra_message_data,
        )
    elif regenerate:
        await asyncio.to_thread(
            storage.replace_last_assistant_message,
            user_id,
            session_id,
            ai_message,
            rag_trace,
        )
    else:
        await asyncio.to_thread(
            storage.append_messages,
            user_id,
            session_id,
            [user_message, ai_message],
            None,
            [None, {"rag_trace": rag_trace}],
        )
