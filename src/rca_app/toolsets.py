from __future__ import annotations

import logging

from .config import AppConfig
from .mcp_toolset import build_mcp_toolset
from .toolset_registry import Toolset

logger = logging.getLogger(__name__)


def build_salesforce_toolset(config: AppConfig) -> Toolset:
    logger.info("Building Salesforce MCP toolset from %s", config.salesforce_mcp_url)
    return build_mcp_toolset(
        name="salesforce",
        description="Salesforce MCP toolset for sales and promotions data.",
        base_url=config.salesforce_mcp_url,
    )


def build_sap_business_one_toolset(config: AppConfig) -> Toolset:
    logger.info("Building SAP Business One MCP toolset from %s", config.sap_mcp_url)
    return build_mcp_toolset(
        name="sap-business-one",
        description="SAP Business One MCP toolset for inventory operations.",
        base_url=config.sap_mcp_url,
    )
