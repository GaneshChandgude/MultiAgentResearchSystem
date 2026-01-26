from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langgraph.graph import StateGraph

from .agents import build_agents, orchestration_agent
from .config import AppConfig
from .memory import setup_memory
from .types import RCAState


@dataclass
class RCAApp:
    config: AppConfig
    store: Any
    checkpointer: Any
    llm: Any
    router_agent: Any
    app: Any


def build_app(config: AppConfig) -> RCAApp:
    memory = setup_memory(config)
    store = memory.store
    checkpointer = memory.checkpointer

    agents = build_agents(config, store, checkpointer)
    router_agent = agents["router_agent"]
    llm = agents["llm"]

    graph = StateGraph(RCAState)
    graph.add_node(
        "orchestration_agent",
        lambda rca_state, config: orchestration_agent(rca_state, config, store, router_agent),
    )
    graph.set_entry_point("orchestration_agent")
    app = graph.compile(checkpointer=checkpointer, store=store)

    return RCAApp(
        config=config,
        store=store,
        checkpointer=checkpointer,
        llm=llm,
        router_agent=router_agent,
        app=app,
    )


def run_rca(app: RCAApp, task: str, user_id: str, query_id: str) -> Dict[str, Any]:
    config = {"configurable": {"user_id": user_id, "thread_id": query_id}}
    rca_state: RCAState = {
        "task": task,
        "output": "",
        "trace": [],
        "history": [],
    }
    return app.app.invoke(rca_state, config)
