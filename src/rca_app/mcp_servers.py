from __future__ import annotations

import logging

from .config import AppConfig
from .inventory_mcp_server import init_inventory_mcp_server
from .sales_mcp_server import init_sales_mcp_server

logger = logging.getLogger(__name__)


def run_salesforce_mcp(config: AppConfig, host: str, port: int) -> None:
    mcp = init_sales_mcp_server(config)
    logger.info("Starting Sales MCP server on %s:%s", host, port)
    mcp.run(transport="sse", host=host, port=port)


def run_sap_business_one_mcp(config: AppConfig, host: str, port: int) -> None:
    mcp = init_inventory_mcp_server(config)
    logger.info("Starting Inventory MCP server on %s:%s", host, port)
    mcp.run(transport="sse", host=host, port=port)
