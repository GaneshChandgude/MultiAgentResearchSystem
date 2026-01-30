from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict

from langgraph.graph import StateGraph

from .agents import build_agents, orchestration_agent
from .config import AppConfig
from .memory import setup_memory
from .observability import build_langfuse_invoke_config
from .types import RCAState

logger = logging.getLogger(__name__)

@dataclass
class RCAApp:
    config: AppConfig
    store: Any
    checkpointer: Any
    llm: Any
    router_agent: Any
    app: Any


def build_app(config: AppConfig) -> RCAApp:
    logger.info("Building RCA application")
    memory = setup_memory(config)
    store = memory.store
    checkpointer = memory.checkpointer

    agents = build_agents(config, store, checkpointer)
    router_agent = agents["router_agent"]
    llm = agents["llm"]

    graph = StateGraph(RCAState)
    graph.add_node(
        "orchestration_agent",
        lambda rca_state, run_config: orchestration_agent(
            rca_state,
            run_config,
            store,
            router_agent,
            app_config=config,
        ),
    )
    graph.set_entry_point("orchestration_agent")
    app = graph.compile(checkpointer=checkpointer, store=store)

    logger.info("RCA application build complete")
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
    observability_config = build_langfuse_invoke_config(
        app.config,
        user_id=user_id,
        query_id=query_id,
        tags=["RCAApp"],
        metadata={"entrypoint": "run_rca", "task_length": len(task)},
    )
    rca_state: RCAState = {
        "task": task,
        "output": "",
        "trace": [],
    }
    logger.info("Running RCA for user_id=%s query_id=%s", user_id, query_id)
    logger.debug("RCA task length=%s", len(task))
    return app.app.invoke(rca_state, {**config, **observability_config})
