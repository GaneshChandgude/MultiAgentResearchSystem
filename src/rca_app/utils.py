from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_core.messages import AIMessage
from langgraph.types import Command

from .config import AppConfig
from .guardrails import apply_output_guardrails, apply_value_guardrails
from .langfuse_prompts import PROMPT_DEFINITIONS, render_prompt

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)

_todo_progress_sink: Callable[[str, List[Dict[str, Any]], Dict[str, Any]], None] | None = None


def register_todo_progress_sink(
    sink: Callable[[str, List[Dict[str, Any]], Dict[str, Any]], None] | None,
) -> None:
    global _todo_progress_sink
    _todo_progress_sink = sink


def extract_json_from_response(response_text: str) -> str:
    match = re.search(r"```json\s*\n([\s\S]*?)\n```", response_text)
    if match:
        return match.group(1).strip()

    match = re.search(r"```json\s*([\s\S]*?)```", response_text)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n?([\s\S]*?)\n?```", response_text)
    if match:
        return match.group(1).strip()

    return response_text.strip()


def process_response(response_content: str, llm=None, app_config: AppConfig | None = None) -> Dict[str, Any]:
    json_decoder_prompt = PROMPT_DEFINITIONS["rca.json_recovery.system"]

    last_exception = None
    logger.debug("Processing LLM response content length=%s", len(response_content))

    for attempt in range(1, 4):
        try:
            content = extract_json_from_response(response_content)
            if isinstance(content, str):
                content = json.loads(content)
            if isinstance(content, str):
                content = json.loads(content)
            logger.debug("Successfully parsed response on attempt %s", attempt)
            return content
        except json.JSONDecodeError as e:
            last_exception = e
            logger.debug("JSON decode failed on attempt %s: %s", attempt, e)
            if llm is None:
                break
            prompt_content = render_prompt(
                app_config,
                name="rca.json_recovery.system",
                fallback=json_decoder_prompt,
                variables={"e": str(e), "response": response_content},
            ) if app_config else json_decoder_prompt.format(e=str(e), response=response_content)
            recovery_prompt = {
                "role": "system",
                "content": prompt_content,
            }
            fixed_response = llm.invoke([recovery_prompt])
            response_content = fixed_response.content

    raise ValueError(f"Model response could not be parsed: {last_exception}")


def _build_tool_error_message(error: Exception, tool_name: str | None) -> str:
    error_text = str(error)
    missing_fields = re.findall(r"([A-Za-z0-9_]+)\s+Field required", error_text)
    if missing_fields:
        fields_list = ", ".join(sorted(set(missing_fields)))
        guidance = (
            f"Missing required fields: {fields_list}. "
            "Please call the tool again with all required inputs. "
        )
        if tool_name == "hypothesis_validation_agent_tool":
            guidance += (
                "Include sales_insights and inventory_insights from prior analysis outputs."
            )
        return f"Tool error: {guidance} ({error_text})"

    return f"Tool error: Please check your input and try again. ({error_text})"


@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        tool_name = request.tool_call.get("name") if request.tool_call else None
        return ToolMessage(
            content=_build_tool_error_message(e, tool_name),
            tool_call_id=request.tool_call["id"],
        )




def _extract_todos_from_tool_call(request: Any, result: Any) -> List[Dict[str, Any]]:
    tool_call = request.tool_call if isinstance(request.tool_call, dict) else {}
    args = tool_call.get("args", {})
    raw_todos = args.get("todos") if isinstance(args, dict) else None

    if not isinstance(raw_todos, list) and isinstance(result, Command) and isinstance(result.update, dict):
        command_todos = result.update.get("todos")
        if isinstance(command_todos, list):
            raw_todos = command_todos

    if not isinstance(raw_todos, list):
        return []

    return [todo for todo in raw_todos if isinstance(todo, dict)]


def _build_todo_progress(todos: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(todos)
    completed = 0
    in_progress = 0

    for todo in todos:
        status = str(todo.get("status", "pending")).strip().lower()
        if status == "completed":
            completed += 1
        elif status == "in_progress":
            in_progress += 1

    percent = int((completed / total) * 100) if total else 0
    return {
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "pending": max(total - completed - in_progress, 0),
        "percent": percent,
        "source": "write_todos",
    }


def _extract_query_id(request: Any) -> str | None:
    tool_call = request.tool_call if isinstance(getattr(request, "tool_call", None), dict) else {}
    tool_args = tool_call.get("args", {})
    if isinstance(tool_args, dict):
        query_id = tool_args.get("query_id")
        if isinstance(query_id, str) and query_id.strip():
            return query_id.strip()

    runtime = getattr(request, "runtime", None)
    runtime_context = getattr(runtime, "context", None)
    if isinstance(runtime_context, dict):
        configurable = runtime_context.get("configurable", {})
        if isinstance(configurable, dict):
            query_id = configurable.get("query_id")
            if isinstance(query_id, str) and query_id.strip():
                return query_id.strip()

    request_config = getattr(request, "config", None)
    if isinstance(request_config, dict):
        configurable = request_config.get("configurable", {})
        if isinstance(configurable, dict):
            query_id = configurable.get("query_id")
            if isinstance(query_id, str) and query_id.strip():
                return query_id.strip()

    if isinstance(getattr(request, "state", None), dict):
        query_id = request.state.get("query_id")
        if isinstance(query_id, str) and query_id.strip():
            return query_id.strip()

        state_configurable = request.state.get("configurable")
        if isinstance(state_configurable, dict):
            query_id = state_configurable.get("query_id")
            if isinstance(query_id, str) and query_id.strip():
                return query_id.strip()

    return None


@wrap_tool_call
def sync_todo_progress(request, handler):
    result = handler(request)

    tool_name = request.tool_call.get("name") if isinstance(request.tool_call, dict) else None
    if tool_name != "write_todos":
        return result

    todos = _extract_todos_from_tool_call(request, result)
    if not todos or not isinstance(request.state, dict):
        return result

    todo_progress = _build_todo_progress(todos)
    request.state["todos"] = todos
    request.state["todo_progress"] = todo_progress

    query_id = _extract_query_id(request)
    if _todo_progress_sink and query_id:
        try:
            _todo_progress_sink(query_id, todos, todo_progress)
        except Exception:
            logger.exception("Failed to persist todo progress for query_id=%s", query_id)

    return result

def make_tool_output_guardrails(config: AppConfig):
    @wrap_tool_call
    def handle_tool_output_guardrails(request, handler):
        result = handler(request)
        if isinstance(result, ToolMessage):
            sanitized = apply_output_guardrails(
                str(getattr(result, "content", "")),
                config=config,
                run_model_guardrails=False,
                enforce_language=False,
                enforce_max_length=False,
            )
            return ToolMessage(content=sanitized, tool_call_id=result.tool_call_id)
        return apply_value_guardrails(result, config=config)

    return handle_tool_output_guardrails


def serialize_messages(msgs: List[Any]) -> List[Dict[str, Any]]:
    cleaned = []
    tool_call_status: Dict[str, str] = {}

    for m in msgs:
        if hasattr(m, "tool_call_id"):
            content = str(getattr(m, "content", ""))
            status = "error" if content.strip().lower().startswith("tool error:") else "success"
            tool_call_status[m.tool_call_id] = status

    for m in msgs:
        entry: Dict[str, Any] = {
            "type": m.__class__.__name__,
            "content": m.content,
        }
        if hasattr(m, "tool_calls") and m.tool_calls:
            entry["tool_calls"] = [
                {
                    "name": tc.get("name"),
                    "args": tc.get("args"),
                    "id": tc.get("id"),
                    "status": tool_call_status.get(tc.get("id"), "unknown"),
                }
                for tc in m.tool_calls
            ]
        if hasattr(m, "tool_call_id"):
            entry["tool_call_id"] = m.tool_call_id

        cleaned.append(entry)

    return cleaned


def filter_tool_messages(messages: List[Any]) -> List[Any]:
    return [
        m
        for m in messages
        if (
            (isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
            or isinstance(m, ToolMessage)
        )
    ]
