from __future__ import annotations

import logging

from .config import AppConfig
from typing import Any, Dict

from mcp.server.fastmcp import FastMCP

from .toolsets import Toolset, build_salesforce_toolset, build_sap_business_one_toolset

logger = logging.getLogger(__name__)


def _tool_input_schema(tool) -> Dict[str, Any]:
    schema = {"type": "object", "properties": {}}
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return schema
    return args_schema.schema()


def _invoke_tool(tool, arguments: Dict[str, Any]):
    if hasattr(tool, "invoke"):
        return tool.invoke(arguments)
    if hasattr(tool, "run"):
        return tool.run(**arguments)
    return tool(**arguments)


def _register_tool(mcp: FastMCP, toolset: Toolset, tool) -> None:
    name = getattr(tool, "name", "unknown")
    description = getattr(tool, "description", "")
    input_schema = _tool_input_schema(tool)

    def handler(**kwargs):
        return _invoke_tool(tool, kwargs)

    handler.__name__ = name
    mcp.tool(name=name, description=description, input_schema=input_schema)(handler)
    logger.debug("Registered tool %s with %s MCP server", name, toolset.name)


def run_mcp_server(toolset: Toolset, host: str, port: int) -> None:
    mcp = FastMCP(toolset.name)
    for tool in toolset.tools:
        _register_tool(mcp, toolset, tool)

    logger.info("Starting %s MCP server on %s:%s", toolset.name, host, port)
    mcp.run(transport="sse", host=host, port=port)


def run_salesforce_mcp(config: AppConfig, host: str, port: int) -> None:
    toolset = build_salesforce_toolset(config)
    run_mcp_server(toolset, host=host, port=port)


def run_sap_business_one_mcp(config: AppConfig, host: str, port: int) -> None:
    toolset = build_sap_business_one_toolset(config)
    run_mcp_server(toolset, host=host, port=port)
