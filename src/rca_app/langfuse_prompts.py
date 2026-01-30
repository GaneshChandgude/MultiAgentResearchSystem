from __future__ import annotations

import base64
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import requests
from urllib3.exceptions import InsecureRequestWarning

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

SALES_ANALYSIS_AGENT_PROMPT = """
You are a Sales Analysis Agent for RCA.

Context (do not repeat, only use for reasoning):
{memory_context}

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
""".strip()

INVENTORY_ANALYSIS_AGENT_PROMPT = """
You are the Inventory RCA Agent.

Context (do not repeat, only use for reasoning):
{memory_context}

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
Produce a final Root Cause Analysis.

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

Create a professional Root Cause Analysis Report.

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
You are analyzing conversations from a supply-chain Root Cause Analysis (RCA) assistant to create episodic memories that will improve future RCA interactions.

Your task is to extract the most useful, reusable insights from the conversation that would help when handling similar RCA scenarios in the future.

Review the conversation and create a memory reflection following these rules:

1. For any field where information is missing or not applicable, use "N/A"
2. Be extremely concise — each string must be one clear, actionable sentence
3. Focus only on information that improves future RCA effectiveness
4. Context_tags must be specific enough to match similar RCA situations but general enough to be reusable

Output valid JSON in exactly this format:
{{
    "context_tags": [               // 2–4 keywords identifying similar RCA scenarios
        string,                     // Use domain-specific terms like "sales_decline", "inventory_stockout", "logistics_delay", "forecast_bias"
        ...
    ],
    "conversation_summary": string, // One sentence describing what RCA problem was addressed and resolved
    "what_worked": string,          // Most effective RCA technique or reasoning strategy used
    "what_to_avoid": string         // Key RCA pitfall or ineffective approach to avoid in future
}}

Do not include any text outside the JSON object in your response.

Here is the prior conversation:

{conversation}
""".strip()

PROCEDURAL_REFLECTION_PROMPT = """
You are extracting PROCEDURAL MEMORY for an RCA agent.

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
Conversation:
{conversation}
""".strip()

SEMANTIC_ABSTRACTION_PROMPT = """
You are building SEMANTIC MEMORY for an RCA agent.

Given multiple episodic RCA reflections, extract generalized,
reusable knowledge that holds across cases.

Rules:
- Do NOT mention specific dates, stores, or conversations
- Focus on patterns, causal relationships, and general truths
- One semantic fact should apply to many future RCA cases

Output ONLY valid JSON in this format:
{{
  "semantic_fact": "string",
  "applicable_context": ["keyword1", "keyword2"],
  "confidence": "low | medium | high"
}}

Episodic memories:
{episodes}
""".strip()

JSON_DECODER_PROMPT = """
You are an expert in resolving JSON decoding errors.

Please review the AI Output (enclosed in triple backticks).

We encountered the following error while loading the AI Output into a JSON object: {e}. Kindly resolve this issue.

AI Output: '''{response}'''

Return ONLY the corrected JSON.
""".strip()


PROMPT_DEFINITIONS: Dict[str, str] = {
    "rca.orchestration.system": ORCHESTRATION_AGENT_PROMPT,
    "rca.hypothesis.system": HYPOTHESIS_AGENT_PROMPT,
    "rca.sales.system": SALES_ANALYSIS_AGENT_PROMPT,
    "rca.inventory.system": INVENTORY_ANALYSIS_AGENT_PROMPT,
    "rca.validation.system": VALIDATION_AGENT_PROMPT,
    "rca.root_cause.system": ROOT_CAUSE_AGENT_PROMPT,
    "rca.report.system": REPORT_AGENT_PROMPT,
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
    verify = _langfuse_verify_setting(config)
    _maybe_disable_insecure_request_warnings(config)

    try:
        response = requests.get(
            url, headers=headers, params=params, timeout=timeout_s, verify=verify
        )
    except requests.RequestException as exc:
        _log_ssl_help(exc, config, "prompt fetch")
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
