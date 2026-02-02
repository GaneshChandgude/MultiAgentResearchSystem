from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict

from langchain_core.runnables import config as runnable_config
from langgraph.graph import StateGraph

from .agents import build_agents, orchestration_agent
from .config import AppConfig
from .memory import setup_memory
from .observability import build_langfuse_invoke_config
from .types import RCAState

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


def _hydrate_history_from_checkpoint(
    rca_state: RCAState,
    checkpointer: Any,
    config: Dict[str, Any] | None,
) -> None:
    if rca_state.get("history"):
        return
    if not config:
        return
    configurable = config.get("configurable", {})
    if "thread_id" not in configurable:
        return
    checkpoint_tuple = checkpointer.get_tuple({"configurable": dict(configurable)})
    if not checkpoint_tuple:
        return
    history = checkpoint_tuple.checkpoint.get("channel_values", {}).get("history")
    if history:
        rca_state["history"] = history
        logger.debug("Hydrated history from checkpointer entries=%s", len(history))

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
    def run_orchestration(rca_state, runtime_config=None):
        if runtime_config is None:
            runtime_config = runnable_config.ensure_config()
        _hydrate_history_from_checkpoint(rca_state, checkpointer, runtime_config)
        return orchestration_agent(
            rca_state,
            runtime_config,
            store,
            router_agent,
            app_config=config,
        )
    graph.add_node(
        "orchestration_agent",
        run_orchestration,
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


def run_rca(
    app: RCAApp,
    task: str,
    user_id: str,
    query_id: str,
    thread_id: str | None = None,
) -> Dict[str, Any]:
    resolved_thread_id = thread_id or query_id
    config = {
        "configurable": {
            "user_id": user_id,
            "thread_id": resolved_thread_id,
            "query_id": query_id,
        }
    }
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
    _hydrate_history_from_checkpoint(
        rca_state,
        app.checkpointer,
        {"configurable": {"thread_id": resolved_thread_id, "user_id": user_id}},
    )
    logger.info(
        "Running RCA for user_id=%s query_id=%s thread_id=%s",
        user_id,
        query_id,
        resolved_thread_id,
    )
    logger.debug("RCA task length=%s", len(task))
    return app.app.invoke(rca_state, {**config, **observability_config})
