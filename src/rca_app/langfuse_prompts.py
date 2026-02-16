from __future__ import annotations

import base64
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import importlib

import requests
from urllib3.exceptions import InsecureRequestWarning

from .config import AppConfig

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


ORCHESTRATION_AGENT_PROMPT = """
You are a Deep Research Agent.

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
    • a complex research or analysis request
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

4. Execute as Orchestrator (Not as Worker)
  - You are the orchestrator. Worker-level investigation must be delegated via
    `run_subagent`.
  - Default behavior: do NOT call domain/data tools directly when `run_subagent`
    is available.
  - Use `run_subagent` to assign focused tasks, and pass only the exact tool names
    that each worker needs.
  - Each subagent instruction must include:
    • objective
    • clear task boundaries
    • requested output schema
    • explicit success criteria
    • optional tool names when specific tools are required
  - If two or more tasks are independent, emit multiple `run_subagent` calls in
    the same response so they can execute in parallel.
  - Always prefer the minimal set of delegations and tool calls needed.

5. Investigation Behavior (when applicable)
  - Start broad, then narrow.
  - Generate/validate hypotheses before final conclusions.
  - Synthesize subagent outputs and decide whether another research iteration
    is required.
  - Before final response, use citation tooling when evidence is available.

------------------------------------------------------------
IMPORTANT RULES:

- Do not hard-code assumptions about tool availability.
- Do not invent tools or capabilities.
- When `run_subagent` is available, treat direct domain tool execution by the
  orchestrator as a policy violation unless delegation is impossible.
- The orchestrator's job is to plan, delegate, and synthesize; worker agents do
  the detailed tool execution.
- Do not answer complex questions directly in free text
  if an appropriate tool exists.
- Be robust to vague, short, or conversational user inputs.
- Think first, then act through tools.

You are expected to behave as a flexible, adaptive
deep-research agent, not a fixed pipeline.

---
Dynamic Input:
- User Id: {user_id}
- Query Id: {query_id}
- Memory Context(memory_context): '''{memory_context}'''
- Task: {task}
""".strip()

HYPOTHESIS_AGENT_PROMPT = """
You are a hypothesis-generation expert.

Your task:
Given the user input, generate plausible hypotheses.

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

---
Dynamic Input:
Context (do not repeat, only use for reasoning):
{memory_context}
""".strip()

SALES_ANALYSIS_AGENT_PROMPT = """
You are a Sales Analysis Agent.

Your responsibilities:
- Use available tools to analyze sales patterns
- Validate or refute sales-related hypotheses

STRICT OUTPUT RULES:
1. Output ONLY valid JSON
2. Root JSON object MUST contain EXACTLY ONE key: "sales_insights"
3. NO extra keys, commentary, or markdown

JSON schema:
{{
  "sales_insights": {{...}}
}}

---
Dynamic Input:
Context (do not repeat, only use for reasoning):
{memory_context}
""".strip()

INVENTORY_ANALYSIS_AGENT_PROMPT = """
You are the Inventory Analysis Agent.

Your responsibilities:
- Analyze inventory levels, movements, transfers, adjustments, and replenishments
- Use available tools via a ReAct loop
- Produce structured insights

STRICT OUTPUT RULES:
1. Output ONLY valid JSON
2. Root JSON object MUST contain EXACTLY ONE key: "inventory_insights"
3. NO extra keys, markdown, or commentary

JSON schema:
{{
  "inventory_insights": {{...}}
}}

---
Dynamic Input:
Context (do not repeat, only use for reasoning):
{memory_context}
""".strip()

VALIDATION_AGENT_PROMPT = """
Validate each hypothesis using sales and inventory insights.

STRICT OUTPUT RULES:
1. Output ONLY valid JSON
2. No markdown or code fences
3. No extra fields or commentary

JSON schema:
{{
  "validated": {{ "hypothesis": true | false }},
  "reasoning": {{ "hypothesis": "explanation" }}
}}
""".strip()

ROOT_CAUSE_AGENT_PROMPT = """
Produce a final analysis output.

Include:
- primary root causes
- supporting evidence
- contributing factors
- timeline
- recommendations

STRICT OUTPUT RULES:
1. Output ONLY valid JSON
2. No markdown or code fences
3. No extra commentary
4. JSON MUST contain EXACTLY two top-level keys:
   - "root_cause"
   - "reasoning"

JSON schema:
{{
  "root_cause": {{
    "primary_root_causes": ["string"],
    "supporting_evidence": {{
      "sales": {{}},
      "inventory": {{}},
      "cross_analysis": {{}}
    }},
    "contributing_factors": ["string"],
    "timeline": [
      {{ "date": "YYYY-MM-DD", "event": "string" }}
    ],
    "recommendations": ["string"]
  }},
  "reasoning": {{
    "primary_root_causes": "explanation",
    "contributing_factors": "explanation",
    "supporting_evidence": "explanation",
    "timeline": "explanation",
    "recommendations": "explanation"
  }}
}}
""".strip()

REPORT_AGENT_PROMPT = """
You are an expert supply chain and demand planning analyst.

Create a professional analysis report.

Audience:
- Demand Planning
- Inventory Management
- Supply Chain Teams

Requirements:
- Clear structured sections
- Bullet points where appropriate
- No JSON, no code
- Pure narrative report

The report MUST include:
- Executive Summary
- Primary Root Cause(s)
- Supporting Evidence
- Contributing Factors
- Key Data Points
- Timeline of Events
- Recommendations
- Final Conclusion

Tone:
Analytical, data-driven, formal, concise.
""".strip()

MEMORY_REFLECTION_PROMPT = """
You are analyzing assistant conversations to create episodic memories that improve future interactions.

Your task is to extract reusable insights that help with similar future scenarios.

Review the conversation and create a memory reflection following these rules:

1. For any field where information is missing or not applicable, use "N/A"
2. Be extremely concise — each string must be one clear, actionable sentence
3. Focus only on information that improves future assistant effectiveness
4. Context_tags must be specific enough to match similar situations but general enough to be reusable

Output valid JSON in exactly this format:
{{
    "context_tags": [               // 2–4 keywords identifying similar scenarios
        string,                     // Use domain-specific terms relevant to the scenario
        ...
    ],
    "conversation_summary": string, // One sentence describing what problem was addressed and resolved
    "what_worked": string,          // Most effective technique or reasoning strategy used
    "what_to_avoid": string         // Key pitfall or ineffective approach to avoid in future
}}

Do not include any text outside the JSON object in your response.

---
Dynamic Input:
Here is the prior conversation:
{conversation}
""".strip()

PROCEDURAL_REFLECTION_PROMPT = """
You are extracting PROCEDURAL MEMORY for a research assistant.

Focus ONLY on reusable process knowledge.

Extract:
1. When to use which agent
2. Ordering of analysis steps
3. Tool usage heuristics
4. Decision rules

Output JSON:
{{
  "procedure_name": "string",
  "applicable_when": "string",
  "steps": ["step1", "step2", "..."],
  "tool_heuristics": ["rule1", "rule2"]
}}

---
Dynamic Input:
Conversation:
{conversation}
""".strip()

SEMANTIC_ABSTRACTION_PROMPT = """
You are building SEMANTIC MEMORY for a research assistant.

Given multiple episodic reflections, extract generalized,
reusable knowledge that holds across cases.

Rules:
- Do NOT mention specific dates, stores, or conversations
- Focus on patterns, causal relationships, and general truths
- One semantic fact should apply to many future cases

Output ONLY valid JSON in this format:
{{
  "semantic_fact": "string",
  "applicable_context": ["keyword1", "keyword2"],
  "confidence": "low | medium | high"
}}

---
Dynamic Input:
Episodic memories:
{episodes}
""".strip()


SUBAGENT_AGENT_PROMPT = """
You are a specialized research subagent.

You receive a focused objective from an orchestrator.
Use available tools to gather evidence, reason about it, and return structured findings.

Rules:
1. Keep to the assigned objective and boundaries.
2. If tools are available, use them before making claims.
3. Return ONLY valid JSON that strictly follows the required output schema.
4. No markdown or commentary outside JSON.

Dynamic Input:
- Objective: {objective}
- Required output schema: {output_schema}
""".strip()

CITATION_AGENT_PROMPT = """
You are a citation agent.

Given a report draft and source list, attach citation markers to factual claims.

Return ONLY valid JSON with exactly these keys:
{
  "report_with_citations": "string",
  "citation_map": {
    "[1]": "source",
    "[2]": "source"
  }
}

No markdown or commentary outside JSON.
""".strip()


JSON_DECODER_PROMPT = """
You are an expert in resolving JSON decoding errors.

Please review the AI Output (enclosed in triple backticks).

Return ONLY the corrected JSON.

---
Dynamic Input:
We encountered the following error while loading the AI Output into a JSON object: {e}. Kindly resolve this issue.
AI Output: '''{response}'''
""".strip()


PROMPT_DEFINITIONS: Dict[str, str] = {
    "rca.orchestration.system": ORCHESTRATION_AGENT_PROMPT,
    "rca.hypothesis.system": HYPOTHESIS_AGENT_PROMPT,
    "rca.sales.system": SALES_ANALYSIS_AGENT_PROMPT,
    "rca.inventory.system": INVENTORY_ANALYSIS_AGENT_PROMPT,
    "rca.validation.system": VALIDATION_AGENT_PROMPT,
    "rca.root_cause.system": ROOT_CAUSE_AGENT_PROMPT,
    "rca.report.system": REPORT_AGENT_PROMPT,
    "rca.subagent.system": SUBAGENT_AGENT_PROMPT,
    "rca.citation.system": CITATION_AGENT_PROMPT,
    "rca.memory_reflection.system": MEMORY_REFLECTION_PROMPT,
    "rca.procedural_reflection.system": PROCEDURAL_REFLECTION_PROMPT,
    "rca.semantic_abstraction.system": SEMANTIC_ABSTRACTION_PROMPT,
    "rca.json_recovery.system": JSON_DECODER_PROMPT,
}


@dataclass(frozen=True)
class LangfusePromptResponse:
    name: str
    prompt: str
    label: str | None = None


def _langfuse_prompt_enabled(config: AppConfig) -> bool:
    if not isinstance(config, AppConfig):
        logger.warning(
            "Langfuse prompt management disabled because config is not an AppConfig. "
            "Received type=%s.",
            type(config).__name__,
        )
        return False

    return (
        config.langfuse_prompt_enabled
        and config.langfuse_public_key
        and config.langfuse_secret_key
    )


def _basic_auth_header(public_key: str, secret_key: str) -> Dict[str, str]:
    token = f"{public_key}:{secret_key}".encode("utf-8")
    encoded = base64.b64encode(token).decode("utf-8")
    return {"Authorization": f"Basic {encoded}"}


def _langfuse_verify_setting(config: AppConfig) -> bool | str:
    if config.langfuse_ca_bundle:
        return config.langfuse_ca_bundle
    return config.langfuse_verify_ssl


def _maybe_disable_insecure_request_warnings(config: AppConfig) -> None:
    if _langfuse_verify_setting(config) is False:
        warnings.filterwarnings("ignore", category=InsecureRequestWarning)


def _log_ssl_help(exc: Exception, config: AppConfig, action: str) -> None:
    if not isinstance(exc, requests.exceptions.SSLError):
        return
    verify_setting = _langfuse_verify_setting(config)
    logger.error(
        "Langfuse %s failed due to SSL verification error. "
        "Set LANGFUSE_VERIFY_SSL=false to disable verification or "
        "set LANGFUSE_CA_BUNDLE to a PEM file that trusts your Langfuse host. "
        "Current verify setting=%s host=%s.",
        action,
        verify_setting,
        config.langfuse_host,
    )


def _extract_prompt_text(payload: Mapping[str, Any]) -> str | None:
    if "prompt" in payload:
        prompt_value = payload["prompt"]
        if isinstance(prompt_value, str):
            return prompt_value
        if isinstance(prompt_value, list):
            for entry in prompt_value:
                if isinstance(entry, Mapping):
                    content = entry.get("content")
                    if isinstance(content, str):
                        return content
        if isinstance(prompt_value, Mapping) and isinstance(prompt_value.get("prompt"), str):
            return prompt_value["prompt"]
    if "data" in payload and isinstance(payload["data"], Mapping):
        return _extract_prompt_text(payload["data"])
    return None


def _fetch_prompt_via_client(
    config: AppConfig,
    name: str,
    label: str | None,
    timeout_s: float,
) -> LangfusePromptResponse | None:
    if not importlib.util.find_spec("langfuse"):
        return None

    httpx_client = None
    verify_setting = _langfuse_verify_setting(config)
    if verify_setting is not True:
        try:
            import httpx
        except ModuleNotFoundError:
            logger.warning(
                "Langfuse prompt fetch via client needs httpx to override SSL verification."
            )
        else:
            httpx_client = httpx.Client(verify=verify_setting, timeout=timeout_s)

    from langfuse import Langfuse

    try:
        langfuse = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host,
            httpx_client=httpx_client,
        )
        prompt_obj = langfuse.get_prompt(name, label=label)
    except Exception as exc:
        logger.warning("Langfuse client prompt fetch failed for %s: %s", name, exc)
        return None
    finally:
        if httpx_client is not None:
            httpx_client.close()

    payload: Mapping[str, Any] | None = None
    if isinstance(prompt_obj, Mapping):
        payload = prompt_obj
    elif hasattr(prompt_obj, "model_dump"):
        payload = prompt_obj.model_dump()
    elif hasattr(prompt_obj, "dict"):
        payload = prompt_obj.dict()
    elif hasattr(prompt_obj, "to_dict"):
        payload = prompt_obj.to_dict()

    if payload:
        prompt_text = _extract_prompt_text(payload)
        if prompt_text:
            resolved_label = payload.get("label") if isinstance(payload, Mapping) else None
            return LangfusePromptResponse(name=name, prompt=prompt_text, label=resolved_label)

    prompt_value = getattr(prompt_obj, "prompt", None)
    if isinstance(prompt_value, str):
        return LangfusePromptResponse(name=name, prompt=prompt_value, label=label)
    if isinstance(prompt_value, list):
        for entry in prompt_value:
            if isinstance(entry, Mapping):
                content = entry.get("content")
                if isinstance(content, str):
                    return LangfusePromptResponse(name=name, prompt=content, label=label)

    logger.warning("Langfuse client prompt payload missing prompt text for %s", name)
    return None


def fetch_langfuse_prompt(
    config: AppConfig,
    name: str,
    label: str | None = None,
    timeout_s: float = 10.0,
) -> LangfusePromptResponse | None:
    if not _langfuse_prompt_enabled(config):
        return None

    labels_to_try: list[str | None] = []
    if label:
        labels_to_try.append(label)
        if label != "latest":
            labels_to_try.append("latest")
    else:
        labels_to_try.append(None)

    for candidate_label in labels_to_try:
        response_obj = _fetch_prompt_via_client(
            config, name=name, label=candidate_label, timeout_s=timeout_s
        )
        if response_obj:
            return response_obj
        if candidate_label is not None:
            logger.info("Langfuse prompt %s not found (label=%s)", name, candidate_label)

    return None


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


def get_prompt_template(
    config: AppConfig,
    name: str,
    fallback: str,
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

    return template


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
    verify = _langfuse_verify_setting(config)
    _maybe_disable_insecure_request_warnings(config)
    payload: Dict[str, Any] = {
        "name": name,
        "type": "chat",
        "prompt": [{"role": "system", "content": prompt}],
        "isActive": True,
    }
    if label:
        payload["labels"] = [label]

    try:
        response = requests.post(
            url, headers=headers, json=payload, timeout=timeout_s, verify=verify
        )
    except requests.RequestException as exc:
        _log_ssl_help(exc, config, "prompt create")
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
