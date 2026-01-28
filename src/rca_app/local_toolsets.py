from __future__ import annotations

import logging

from .config import AppConfig
from .toolset_registry import Toolset
from .tools_inventory import build_inventory_tools
from .tools_sales import build_sales_tools

logger = logging.getLogger(__name__)


def build_local_salesforce_toolset(config: AppConfig) -> Toolset:
    logger.info("Building local Salesforce toolset")
    tools = build_sales_tools(config)
    return Toolset(
        name="salesforce",
        tools=tools,
        description="Salesforce MCP toolset for sales and promotions data.",
    )


def build_local_sap_business_one_toolset(config: AppConfig) -> Toolset:
    logger.info("Building local SAP Business One toolset")
    tools = build_inventory_tools(config)
    return Toolset(
        name="sap-business-one",
        tools=tools,
        description="SAP Business One MCP toolset for inventory operations.",
    )
