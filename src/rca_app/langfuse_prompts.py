from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import requests

from .config import AppConfig

logger = logging.getLogger(__name__)


ORCHESTRATION_AGENT_PROMPT = """
You are a Deep Research Agent.

Task: {task}

User Id: {user_id}

Query Id: {query_id}

Use the following sementic abstract + procedural + episodic + conversation context:
Memory Context(memory_context):'''{memory_context}'''

Your role is to analyze the user's input, determine the appropriate
research or response strategy, and use the available tools to resolve
the request.

The set of tools available to you may change dynamically.
You must infer what each tool does from its description.

------------------------------------------------------------
CORE RESPONSIBILITIES:

1. Understand User Intent
  - The user input may be:
    • a greeting or help request (e.g., "hi", "hello", "help")
    • a general question
    • a root cause analysis or supply chain investigation
  - Do not assume the input is analytical.

2. Decide the Level of Depth Required
  - If the input can be addressed with a simple explanation or response,
    prefer a lightweight approach.
  - If the input requires investigation, reasoning, or analysis,
    proceed with deep research behavior.

3. Create an Internal Plan
  - Before calling any tool, determine:
    • what information is missing
    • what needs to be discovered or generated
    • whether memory or prior context is relevant
  - The plan does not need to be shown unless required by a tool.

4. Execute Using Tools
  - Use the **todo's** tools to carry out the plan.
  - Choose tools based on their descriptions, not their names.
  - You may call multiple tools if necessary.
  - Always prefer the minimal set of tool calls needed.

5. RCA-Specific Behavior (when applicable)
  - When the task involves diagnosing causes of a problem:
    • avoid jumping to conclusions
    • favor hypothesis generation before validation
    • rely on state, memory, and evidence

------------------------------------------------------------
IMPORTANT RULES:

- Do not hard-code assumptions about tool availability.
- Do not invent tools or capabilities.
- Do not answer complex questions directly in free text
  if an appropriate tool exists.
- Be robust to vague, short, or conversational user inputs.
- Think first, then act through tools.

You are expected to behave as a flexible, adaptive
deep-research agent, not a fixed pipeline.
""".strip()

HYPOTHESIS_AGENT_PROMPT = """
You are an RCA hypothesis-generation expert.

Context (do not repeat, only use for reasoning):
{memory_context}

Your task:
Given the user input, generate possible root-cause hypotheses.

STRICT OUTPUT RULES:
1. Output **only valid JSON**.
2. Root JSON object must have exactly two fields:
   - "hypotheses": an array of **plain strings**.
   - "reasoning": a string explaining how the hypotheses were generated.
3. No markdown or code fences.
4. No extra commentary or fields.

JSON schema:
{{
  "hypotheses": ["...", "..."],
  "reasoning": "..."
}}
""".strip()


PROMPT_DEFINITIONS: Dict[str, str] = {
    "rca.orchestration.system": ORCHESTRATION_AGENT_PROMPT,
    "rca.hypothesis.system": HYPOTHESIS_AGENT_PROMPT,
}


@dataclass(frozen=True)
class LangfusePromptResponse:
    name: str
    prompt: str
    label: str | None = None


def _langfuse_prompt_enabled(config: AppConfig) -> bool:
    return (
        config.langfuse_prompt_enabled
        and config.langfuse_public_key
        and config.langfuse_secret_key
    )


def _basic_auth_header(public_key: str, secret_key: str) -> Dict[str, str]:
    token = f"{public_key}:{secret_key}".encode("utf-8")
    encoded = base64.b64encode(token).decode("utf-8")
    return {"Authorization": f"Basic {encoded}"}


def _extract_prompt_text(payload: Mapping[str, Any]) -> str | None:
    if "prompt" in payload:
        prompt_value = payload["prompt"]
        if isinstance(prompt_value, str):
            return prompt_value
        if isinstance(prompt_value, Mapping) and isinstance(prompt_value.get("prompt"), str):
            return prompt_value["prompt"]
    if "data" in payload and isinstance(payload["data"], Mapping):
        return _extract_prompt_text(payload["data"])
    return None


def fetch_langfuse_prompt(
    config: AppConfig,
    name: str,
    label: str | None = None,
    timeout_s: float = 10.0,
) -> LangfusePromptResponse | None:
    if not _langfuse_prompt_enabled(config):
        return None

    url = f"{config.langfuse_host.rstrip('/')}/api/public/prompts/{name}"
    params = {"label": label} if label else None
    headers = _basic_auth_header(config.langfuse_public_key, config.langfuse_secret_key)

    try:
        response = requests.get(url, headers=headers, params=params, timeout=timeout_s)
    except requests.RequestException as exc:
        logger.warning("Failed to reach Langfuse prompt API for %s: %s", name, exc)
        return None

    if response.status_code == 404:
        logger.info("Langfuse prompt %s not found", name)
        return None
    if response.status_code >= 400:
        logger.warning(
            "Langfuse prompt fetch failed name=%s status=%s body=%s",
            name,
            response.status_code,
            response.text,
        )
        return None

    payload = response.json()
    prompt_text = _extract_prompt_text(payload)
    if not prompt_text:
        logger.warning("Langfuse prompt payload missing prompt text for %s", name)
        return None

    resolved_label = payload.get("label") if isinstance(payload, Mapping) else None
    return LangfusePromptResponse(name=name, prompt=prompt_text, label=resolved_label)


def render_prompt(
    config: AppConfig,
    name: str,
    fallback: str,
    variables: Mapping[str, Any],
    label: str | None = None,
) -> str:
    template = fallback
    if _langfuse_prompt_enabled(config):
        response = fetch_langfuse_prompt(
            config, name=name, label=label or config.langfuse_prompt_label
        )
        if response:
            template = response.prompt
            logger.debug("Using Langfuse prompt %s (label=%s)", name, response.label)

    try:
        return template.format(**variables)
    except KeyError as exc:
        logger.warning(
            "Prompt rendering failed for %s due to missing key %s; falling back",
            name,
            exc,
        )
    except ValueError as exc:
        logger.warning("Prompt rendering failed for %s: %s; falling back", name, exc)

    return fallback.format(**variables)


def ensure_langfuse_prompt(
    config: AppConfig,
    name: str,
    prompt: str,
    label: str | None = None,
    timeout_s: float = 10.0,
) -> bool:
    if not _langfuse_prompt_enabled(config):
        logger.info("Langfuse prompt sync skipped; prompt management disabled.")
        return False

    existing = fetch_langfuse_prompt(config, name=name, label=label)
    if existing:
        logger.info("Langfuse prompt %s already exists; skipping create.", name)
        return False

    url = f"{config.langfuse_host.rstrip('/')}/api/public/prompts"
    headers = _basic_auth_header(config.langfuse_public_key, config.langfuse_secret_key)
    payload: Dict[str, Any] = {"name": name, "prompt": prompt}
    if label:
        payload["labels"] = [label]

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    except requests.RequestException as exc:
        logger.warning("Failed to create Langfuse prompt %s: %s", name, exc)
        return False

    if response.status_code >= 400:
        logger.warning(
            "Langfuse prompt create failed name=%s status=%s body=%s",
            name,
            response.status_code,
            response.text,
        )
        return False

    logger.info("Langfuse prompt %s created successfully.", name)
    return True


def sync_prompt_definitions(config: AppConfig, label: str | None = None) -> int:
    synced = 0
    for name, template in PROMPT_DEFINITIONS.items():
        if ensure_langfuse_prompt(config, name=name, prompt=template, label=label):
            synced += 1
    return synced
