from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_core.messages import AIMessage

from .config import AppConfig
from .guardrails import apply_output_guardrails, apply_value_guardrails
from .langfuse_prompts import PROMPT_DEFINITIONS, render_prompt

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


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


def make_tool_output_guardrails(config: AppConfig):
    @wrap_tool_call
    def handle_tool_output_guardrails(request, handler):
        result = handler(request)
        if isinstance(result, ToolMessage):
            sanitized = apply_output_guardrails(str(getattr(result, "content", "")), config=config)
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
