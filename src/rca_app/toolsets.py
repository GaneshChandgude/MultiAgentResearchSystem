from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List

from .config import AppConfig
from .tools_inventory import build_inventory_tools
from .tools_sales import build_sales_tools

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Toolset:
    name: str
    tools: List[object]
    description: str

    def get_tool(self, tool_name: str):
        for tool in self.tools:
            if getattr(tool, "name", None) == tool_name:
                return tool
        raise KeyError(f"Tool '{tool_name}' not found in toolset '{self.name}'.")


def build_salesforce_toolset(config: AppConfig) -> Toolset:
    logger.info("Building Salesforce toolset")
    tools = build_sales_tools(config)
    return Toolset(
        name="salesforce",
        tools=tools,
        description="Salesforce MCP toolset for sales and promotions data.",
    )


def build_sap_business_one_toolset(config: AppConfig) -> Toolset:
    logger.info("Building SAP Business One toolset")
    tools = build_inventory_tools(config)
    return Toolset(
        name="sap-business-one",
        tools=tools,
        description="SAP Business One MCP toolset for inventory operations.",
    )


def find_tool(toolsets: Iterable[Toolset], tool_name: str):
    for toolset in toolsets:
        for tool in toolset.tools:
            if getattr(tool, "name", None) == tool_name:
                logger.debug("Resolved tool %s from toolset %s", tool_name, toolset.name)
                return tool
    raise KeyError(f"Tool '{tool_name}' not found in toolsets.")
