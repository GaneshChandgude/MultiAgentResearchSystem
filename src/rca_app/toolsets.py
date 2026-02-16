from __future__ import annotations

import logging
from urllib.parse import urlparse

from .config import AppConfig
from .mcp_toolset import build_mcp_toolset
from .toolset_registry import Toolset

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


def _uniquify_toolset_name(name: str, used_names: set[str]) -> str:
    if name not in used_names:
        used_names.add(name)
        return name
    suffix = 2
    while True:
        candidate = f"{name}-{suffix}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        suffix += 1

def build_salesforce_toolset(config: AppConfig) -> Toolset:
    logger.info("Building Salesforce MCP toolset from %s", config.salesforce_mcp_url)
    try:
        return build_mcp_toolset(
            name="salesforce",
            description="Salesforce MCP toolset for sales and promotions data.",
            base_url=config.salesforce_mcp_url,
        )
    except Exception as exc:
        logger.warning("Skipping Salesforce MCP toolset: %s", exc)
        return Toolset(
            name="salesforce",
            tools=[],
            description="Salesforce MCP toolset unavailable.",
        )


def build_sap_business_one_toolset(config: AppConfig) -> Toolset:
    logger.info("Building SAP Business One MCP toolset from %s", config.sap_mcp_url)
    try:
        return build_mcp_toolset(
            name="sap-business-one",
            description="SAP Business One MCP toolset for inventory operations.",
            base_url=config.sap_mcp_url,
        )
    except Exception as exc:
        logger.warning("Skipping SAP Business One MCP toolset: %s", exc)
        return Toolset(
            name="sap-business-one",
            tools=[],
            description="SAP Business One MCP toolset unavailable.",
        )


def build_user_mcp_toolsets(config: AppConfig) -> list[Toolset]:
    toolsets: list[Toolset] = []
    used_names: set[str] = set()
    for index, server in enumerate(config.mcp_servers):
        if not isinstance(server, dict):
            continue
        if server.get("enabled", True) is False:
            continue
        base_url = str(server.get("base_url", "")).strip()
        if not base_url:
            continue

        configured_name = str(server.get("name", "")).strip()
        if configured_name:
            name = configured_name
        else:
            host = urlparse(base_url).hostname or "server"
            name = f"mcp-{host}-{index + 1}"
        name = _uniquify_toolset_name(name, used_names)
        description = str(server.get("description", "")).strip() or f"User MCP toolset at {base_url}."

        logger.info("Building user MCP toolset name=%s url=%s", name, base_url)
        try:
            toolsets.append(
                build_mcp_toolset(
                    name=name,
                    description=description,
                    base_url=base_url,
                )
            )
        except Exception as exc:
            logger.warning("Skipping user MCP toolset name=%s: %s", name, exc)
    return toolsets
