from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from .config import AppConfig
from .llm import get_embeddings


@dataclass
class MemoryStores:
    store: InMemoryStore
    checkpointer: InMemorySaver


def setup_memory(config: AppConfig) -> MemoryStores:
    embed = get_embeddings(config)
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": embed,
        }
    )
    checkpointer = InMemorySaver()
    return MemoryStores(store=store, checkpointer=checkpointer)


def append_rca_history(state: Dict[str, Any]) -> None:
    if state.get("task"):
        state.setdefault("history", []).append(HumanMessage(content=state["task"]))
    if state.get("output"):
        state.setdefault("history", []).append(AIMessage(content=state["output"]))


def episodic_recall(query: str, store: InMemoryStore, config: Dict[str, Any]):
    namespace = ("episodic", config["configurable"]["user_id"])
    return store.search(namespace, query=query, limit=1)


def procedural_recall(task: str, store: InMemoryStore, config: Dict[str, Any]):
    namespace = ("procedural", config["configurable"]["user_id"])
    return store.search(namespace, query=task, limit=3)


def semantic_recall(query: str, store: InMemoryStore, config: Dict[str, Any], limit: int = 3):
    namespace = ("semantic", config["configurable"]["user_id"])
    results = store.search(namespace, query=query, limit=limit * 2)

    now = time.time()
    decayed = []

    for r in results:
        v = r.value
        last_used = v.get("last_used_at", now)
        age_days = (now - last_used) / 86400

        if age_days > 30 and v.get("confidence") == "high":
            v["confidence"] = "medium"
        elif age_days > 60:
            v["confidence"] = "low"

        decayed.append(r)

    return decayed[:limit]


def build_memory_augmented_prompt(
    query: str,
    state: Dict[str, Any],
    config: Dict[str, Any],
    store: InMemoryStore,
) -> str:
    semantic_memories = semantic_recall(query, store, config, 3)

    if semantic_memories:
        facts = []
        for sm in semantic_memories:
            v = sm.value
            facts.append(f"- {v.get('semantic_fact')} (confidence: {v.get('confidence')})")
        semantic_context = f"\nGeneralized RCA knowledge:\n{chr(10).join(facts)}\n"
    else:
        semantic_context = "No generalized RCA knowledge found."

    procedural_memories = procedural_recall(query, store, config)

    if procedural_memories:
        procedures = []
        for pm in procedural_memories:
            proc = pm.value
            procedures.append(
                f"""
- Procedure: {proc.get("procedure_name", "N/A")}
  Applicable when: {proc.get("applicable_when", "N/A")}
  Steps:
    {chr(10).join([f"    - {s}" for s in proc.get("steps", [])])}
  Tool heuristics:
    {chr(10).join([f"    - {h}" for h in proc.get("tool_heuristics", [])])}
"""
            )

        procedural_context = f"\nRelevant RCA procedures (how to act):\n{''.join(procedures)}\n"
    else:
        procedural_context = "No relevant RCA procedures found."

    episodic_memories = episodic_recall(query, store, config)

    if episodic_memories:
        mem = episodic_memories[0].value
        episodic_context = f"""
Similar past RCA experience:
- Current Conversation Match: {mem.get('conversation', 'N/A')}
- Summary: {mem.get('conversation_summary', 'N/A')}
- What worked: {mem.get('what_worked', 'N/A')}
- What to avoid: {mem.get('what_to_avoid', 'N/A')}
"""
    else:
        episodic_context = "No closely related past RCA experience found."

    history = state.get("history", [])
    formatted_history = []
    for m in history:
        if isinstance(m, HumanMessage):
            formatted_history.append(f"USER: {m.content}")
        elif isinstance(m, AIMessage):
            formatted_history.append(f"ASSISTANT: {m.content}")

    history_context = "\n".join(formatted_history) if formatted_history else "No prior conversation."

    prompt = f"""
You are an RCA assistant with access to memory.

{semantic_context}

{procedural_context}

{episodic_context}

Recent conversation context:
{history_context}

Instructions:
- Follow relevant procedures first
- Use past experiences to avoid known pitfalls
- Use recent conversation context for continuity
"""

    return prompt.strip()


def format_conversation(history: List[BaseMessage]) -> str:
    conversation = []
    for message in history:
        role = ""
        content = ""
        if isinstance(message, (BaseMessage, HumanMessage, AIMessage)):
            role = message.type.upper()
            content = message.content
        conversation.append(f"{role}: {content}")
    return "\n".join(conversation)


def mark_memory_useful(memories: List[Any]) -> None:
    for m in memories:
        m.value["usefulness"] = m.value.get("usefulness", 0) + 1
