from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import AppConfig
from .memory import format_conversation
from .observability import build_langfuse_invoke_config

logger = logging.getLogger(__name__)

REFLECTION_PROMPT_TEMPLATE = """
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
"""

PROCEDURAL_REFLECTION_TEMPLATE = """
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
"""

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
"""


def build_reflection_chain(llm):
    prompt = ChatPromptTemplate.from_template(REFLECTION_PROMPT_TEMPLATE)
    return prompt | llm | JsonOutputParser()


def build_procedural_chain(llm):
    prompt = ChatPromptTemplate.from_template(PROCEDURAL_REFLECTION_TEMPLATE)
    return prompt | llm | JsonOutputParser()


def build_semantic_chain(llm):
    prompt = ChatPromptTemplate.from_template(SEMANTIC_ABSTRACTION_PROMPT)
    return prompt | llm | JsonOutputParser()


def add_episodic_memory(rca_state, config, store, llm, app_config: AppConfig) -> None:
    history = rca_state.get("history")
    if not history:
        logger.debug("Skipping episodic memory; no history found")
        return

    conversation = format_conversation(history)
    reflect = build_reflection_chain(llm)
    observability_config = build_langfuse_invoke_config(
        app_config,
        user_id=config["configurable"]["user_id"],
        query_id=config["configurable"]["thread_id"],
        tags=["MemoryReflection", "Episodic"],
        metadata={"entrypoint": "add_episodic_memory", "history_length": len(history)},
    )
    reflection = reflect.invoke({"conversation": conversation}, config=observability_config)
    reflection["conversation"] = conversation

    store.put(
        namespace=("episodic", config["configurable"]["user_id"]),
        key=f"episodic_rca_{uuid.uuid4().hex}",
        value=reflection,
    )
    logger.info("Episodic memory stored for user_id=%s", config["configurable"]["user_id"])


def add_procedural_memory(rca_state, config, store, llm, app_config: AppConfig) -> None:
    history = rca_state.get("history")
    if not history:
        logger.debug("Skipping procedural memory; no history found")
        return

    conversation = format_conversation(history)
    procedural_reflection = build_procedural_chain(llm)
    observability_config = build_langfuse_invoke_config(
        app_config,
        user_id=config["configurable"]["user_id"],
        query_id=config["configurable"]["thread_id"],
        tags=["MemoryReflection", "Procedural"],
        metadata={"entrypoint": "add_procedural_memory", "history_length": len(history)},
    )
    reflection = procedural_reflection.invoke(
        {"conversation": conversation},
        config=observability_config,
    )

    store.put(
        namespace=("procedural", config["configurable"]["user_id"]),
        key=f"procedural_rca_{uuid.uuid4().hex}",
        value=reflection,
    )
    logger.info("Procedural memory stored for user_id=%s", config["configurable"]["user_id"])


def build_semantic_memory(
    user_id: str,
    query: str,
    store,
    llm,
    app_config: AppConfig,
    min_episodes: int = 3,
) -> Dict[str, Any] | None:
    episodic = store.search(("episodic", user_id), query=query, limit=10)
    if len(episodic) < min_episodes:
        logger.debug("Insufficient episodic memories for semantic reflection; count=%s", len(episodic))
        return None

    episodes_text = []
    for e in episodic:
        v = e.value
        episodes_text.append(
            f"- Summary: {v.get('conversation_summary')}\n"
            f"  Worked: {v.get('what_worked')}\n"
            f"  Avoid: {v.get('what_to_avoid')}"
        )

    semantic_reflection_chain = build_semantic_chain(llm)
    observability_config = build_langfuse_invoke_config(
        app_config,
        user_id=user_id,
        query_id=None,
        tags=["MemoryReflection", "Semantic"],
        metadata={"entrypoint": "build_semantic_memory", "episode_count": len(episodic)},
    )
    semantic = semantic_reflection_chain.invoke(
        {"episodes": "\n".join(episodes_text)},
        config=observability_config,
    )

    if not semantic or not isinstance(semantic, dict):
        return None

    semantic["usefulness"] = 0
    semantic["last_used_at"] = time.time()

    store.put(
        namespace=("semantic", user_id),
        key=f"semantic_{uuid.uuid4().hex}",
        value=semantic,
    )
    logger.info("Semantic memory stored for user_id=%s", user_id)

    return semantic
