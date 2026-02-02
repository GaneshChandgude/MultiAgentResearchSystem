from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import AppConfig
from .langfuse_prompts import PROMPT_DEFINITIONS, get_prompt_template
from .memory import format_conversation
from .observability import build_langfuse_invoke_config

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


def build_reflection_chain(llm, app_config: AppConfig):
    template = get_prompt_template(
        app_config,
        name="rca.memory_reflection.system",
        fallback=PROMPT_DEFINITIONS["rca.memory_reflection.system"],
    )
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | JsonOutputParser()


def build_procedural_chain(llm, app_config: AppConfig):
    template = get_prompt_template(
        app_config,
        name="rca.procedural_reflection.system",
        fallback=PROMPT_DEFINITIONS["rca.procedural_reflection.system"],
    )
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | JsonOutputParser()


def build_semantic_chain(llm, app_config: AppConfig):
    template = get_prompt_template(
        app_config,
        name="rca.semantic_abstraction.system",
        fallback=PROMPT_DEFINITIONS["rca.semantic_abstraction.system"],
    )
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | JsonOutputParser()


def add_episodic_memory(rca_state, config, store, llm, app_config: AppConfig) -> None:
    history = rca_state.get("history")
    if not history:
        logger.debug("Skipping episodic memory; no history found")
        return

    configurable = config.get("configurable", {})
    query_id = configurable.get("query_id", configurable.get("thread_id"))
    conversation = format_conversation(history)
    reflect = build_reflection_chain(llm, app_config)
    observability_config = build_langfuse_invoke_config(
        app_config,
        user_id=configurable.get("user_id"),
        query_id=query_id,
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

    configurable = config.get("configurable", {})
    query_id = configurable.get("query_id", configurable.get("thread_id"))
    conversation = format_conversation(history)
    procedural_reflection = build_procedural_chain(llm, app_config)
    observability_config = build_langfuse_invoke_config(
        app_config,
        user_id=configurable.get("user_id"),
        query_id=query_id,
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

    semantic_reflection_chain = build_semantic_chain(llm, app_config)
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
