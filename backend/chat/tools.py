from __future__ import annotations

from typing import Optional
from contextvars import ContextVar
import requests
from langchain_core.tools import tool

from backend.config import AMAP_WEATHER_API, AMAP_API_KEY
from backend.chat.rag_execution import RagExecutionPolicy, mark_rag_execution_policy
from backend.rag.formatting import format_rag_tool_response
from backend.rag.trace import mark_context_delivery

_LAST_RAG_CONTEXT: ContextVar[dict | None] = ContextVar('_LAST_RAG_CONTEXT', default=None)
_KNOWLEDGE_TOOL_CALLS_THIS_TURN: ContextVar[dict | None] = ContextVar('_KNOWLEDGE_TOOL_CALLS_THIS_TURN', default=None)
_RAG_CONTEXT_FILES_THIS_TURN: ContextVar[list[str]] = ContextVar('_RAG_CONTEXT_FILES_THIS_TURN', default=[])
_RAG_STEP_QUEUE: ContextVar[object | None] = ContextVar('_RAG_STEP_QUEUE', default=None)
_RAG_STEP_LOOP: ContextVar[object | None] = ContextVar('_RAG_STEP_LOOP', default=None)


def _set_last_rag_context(context: dict):
    _last_rag_context_holder()["value"] = context


def _last_rag_context_holder() -> dict:
    holder = _LAST_RAG_CONTEXT.get()
    if holder is None or "value" not in holder:
        holder = {"value": None}
        _LAST_RAG_CONTEXT.set(holder)
    return holder


def _knowledge_tool_call_holder() -> dict:
    holder = _KNOWLEDGE_TOOL_CALLS_THIS_TURN.get()
    if holder is None or "count" not in holder:
        holder = {"count": 0}
        _KNOWLEDGE_TOOL_CALLS_THIS_TURN.set(holder)
    return holder


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    holder = _last_rag_context_holder()
    context = holder.get("value")
    if clear:
        holder["value"] = None
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN.set({"count": 0})


def set_rag_context_files(filenames: Optional[list[str]] = None):
    """Set filenames that should constrain knowledge retrieval for the current turn."""
    from backend.shared.filename_utils import dedupe_filenames
    _RAG_CONTEXT_FILES_THIS_TURN.set(dedupe_filenames(filenames))


def get_rag_context_files() -> list[str]:
    """Return current-turn filename constraints for knowledge retrieval."""
    return list(_RAG_CONTEXT_FILES_THIS_TURN.get())


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    _RAG_STEP_QUEUE.set(queue)
    if queue:
        import asyncio
        try:
            _RAG_STEP_LOOP.set(asyncio.get_running_loop())
        except RuntimeError:
            _RAG_STEP_LOOP.set(asyncio.get_event_loop())
    else:
        _RAG_STEP_LOOP.set(None)


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    queue = _RAG_STEP_QUEUE.get()
    loop = _RAG_STEP_LOOP.get()
    if queue is not None and loop is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, step)
        except Exception:
            pass


def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取天气信息"""
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    call_holder = _knowledge_tool_call_holder()
    calls = int(call_holder.get("count", 0) or 0)
    if calls >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    call_holder["count"] = calls + 1

    from backend.rag.pipeline import run_rag_graph

    rag_result = run_rag_graph(query, context_files=get_rag_context_files())

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    context = rag_result.get("context", "") if isinstance(rag_result, dict) else ""
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        rag_trace = mark_context_delivery(
            rag_trace,
            delivery_mode="tool_response",
            context=context,
            docs=docs,
        ) or {}
        rag_trace = mark_rag_execution_policy(
            rag_trace,
            policy=RagExecutionPolicy.OPTIONAL_TOOL,
            delivery_mode="tool_response",
            policy_reason="tool_invoked",
        ) or {}
        _set_last_rag_context({"rag_trace": rag_trace, "context": context})

    return format_rag_tool_response(docs, context=context)
