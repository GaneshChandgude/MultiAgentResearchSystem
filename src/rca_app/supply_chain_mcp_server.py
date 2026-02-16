from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .agents import (
    build_hypothesis_tool,
    build_inventory_analysis_tool,
    build_report_tool,
    build_root_cause_tool,
    build_sales_analysis_tool,
    build_validation_tool,
)
from .config import AppConfig
from .memory import setup_memory
from .toolsets import build_salesforce_toolset, build_sap_business_one_toolset

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)

supply_chain_mcp_server = FastMCP("SupplyChainMCPServer")
_specialist_tools: dict[str, Any] = {}


def _require_tools_initialized() -> dict[str, Any]:
    if not _specialist_tools:
        raise RuntimeError(
            "Supply Chain MCP server not initialized. Call init_supply_chain_mcp_server(config)."
        )
    return _specialist_tools


def _find_tool_by_name(tools: list[Any], name: str) -> Any | None:
    for tool in tools:
        if getattr(tool, "name", "") == name:
            return tool
    return None


@supply_chain_mcp_server.tool()
def hypothesis_tool(task: str, user_id: str, query_id: str, memory_context: str = "") -> Dict[str, Any]:
    """Generate plausible hypotheses for the given RCA task."""
    tools = _require_tools_initialized()
    return tools["hypothesis_tool"].invoke(
        {
            "task": task,
            "user_id": user_id,
            "query_id": query_id,
            "memory_context": memory_context,
        }
    )


@supply_chain_mcp_server.tool()
def sales_tool(
    task: str,
    hypotheses: List[str],
    user_id: str,
    query_id: str,
    memory_context: str = "",
) -> Dict[str, Any]:
    """Validate hypotheses using sales and promotions data."""
    tools = _require_tools_initialized()
    return tools["sales_tool"].invoke(
        {
            "task": task,
            "hypotheses": hypotheses,
            "user_id": user_id,
            "query_id": query_id,
            "memory_context": memory_context,
        }
    )


@supply_chain_mcp_server.tool()
def inventory_tool(
    task: str,
    hypotheses: List[str],
    user_id: str,
    query_id: str,
    memory_context: str = "",
) -> Dict[str, Any]:
    """Validate hypotheses using inventory and replenishment data."""
    tools = _require_tools_initialized()
    return tools["inventory_tool"].invoke(
        {
            "task": task,
            "hypotheses": hypotheses,
            "user_id": user_id,
            "query_id": query_id,
            "memory_context": memory_context,
        }
    )


@supply_chain_mcp_server.tool()
def validation_tool(
    hypotheses: List[str],
    user_id: str,
    query_id: str,
    sales_insights: Optional[Dict[str, Any]] = None,
    inventory_insights: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cross-validate hypotheses using available specialist insights."""
    tools = _require_tools_initialized()
    return tools["validation_tool"].invoke(
        {
            "hypotheses": hypotheses,
            "user_id": user_id,
            "query_id": query_id,
            "sales_insights": sales_insights,
            "inventory_insights": inventory_insights,
        }
    )


@supply_chain_mcp_server.tool()
def root_cause_tool(
    validated_hypotheses: Dict[str, bool],
    sales_insights: Dict[str, Any],
    inventory_insights: Dict[str, Any],
    user_id: str,
    query_id: str,
) -> Dict[str, Any]:
    """Synthesize validated findings into a structured root cause output."""
    tools = _require_tools_initialized()
    return tools["root_cause_tool"].invoke(
        {
            "validated_hypotheses": validated_hypotheses,
            "sales_insights": sales_insights,
            "inventory_insights": inventory_insights,
            "user_id": user_id,
            "query_id": query_id,
        }
    )


@supply_chain_mcp_server.tool()
def report_tool(root_cause: str, reasoning: str, user_id: str, query_id: str) -> Dict[str, Any]:
    """Generate a final human-readable RCA report."""
    tools = _require_tools_initialized()
    return tools["report_tool"].invoke(
        {
            "root_cause": root_cause,
            "reasoning": reasoning,
            "user_id": user_id,
            "query_id": query_id,
        }
    )


def init_supply_chain_mcp_server(config: AppConfig) -> FastMCP:
    memory = setup_memory(config)
    store = memory.store
    checkpointer = memory.checkpointer

    logger.debug("Initializing SupplyChainMCPServer with specialist model %s", config.specialist_model)

    from .llm import get_specialist_llm_model

    base_specialist_llm = get_specialist_llm_model(config)
    parallel_specialist_llm = base_specialist_llm.bind(parallel_tool_calls=True)

    sales_tools = build_salesforce_toolset(config).tools
    inventory_tools = build_sap_business_one_toolset(config).tools
    promo_tool = _find_tool_by_name(sales_tools, "get_promo_period")

    hypothesis = build_hypothesis_tool(config, store, checkpointer, parallel_specialist_llm)
    sales, _ = build_sales_analysis_tool(
        config,
        store,
        checkpointer,
        parallel_specialist_llm,
        sales_tools,
    )
    inventory = build_inventory_analysis_tool(
        config,
        store,
        checkpointer,
        parallel_specialist_llm,
        inventory_tools,
        promo_tool,
    )
    validation = build_validation_tool(config, store, checkpointer, parallel_specialist_llm)
    root_cause = build_root_cause_tool(config, store, checkpointer, base_specialist_llm)
    report = build_report_tool(config, store, checkpointer, base_specialist_llm)

    _specialist_tools.clear()
    _specialist_tools.update(
        {
            "hypothesis_tool": hypothesis,
            "sales_tool": sales,
            "inventory_tool": inventory,
            "validation_tool": validation,
            "root_cause_tool": root_cause,
            "report_tool": report,
        }
    )
    logger.info("SupplyChainMCPServer initialized with %s specialist tools", len(_specialist_tools))
    return supply_chain_mcp_server
