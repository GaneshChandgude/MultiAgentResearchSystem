from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from .config import AppConfig
from .data import load_sales

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)

sales_mcp_server = FastMCP("SalesMCPServer")
_sales_config: AppConfig | None = None


def _get_config() -> AppConfig:
    if _sales_config is None:
        raise RuntimeError("Sales MCP server not initialized. Call init_sales_mcp_server(config).")
    return _sales_config


@sales_mcp_server.tool()
def get_daily_sales():
    """Return daily aggregated sales by store."""
    logger.debug("get_daily_sales invoked")
    config = _get_config()
    df = load_sales(config)
    daily = (
        df.groupby(["transaction_date", "store_id", "store_name"], as_index=False)[
            "quantity_sold"
        ]
        .sum()
        .sort_values(["transaction_date", "store_id"])
    )
    records = daily.to_dict(orient="records")
    logger.debug("get_daily_sales returning %s records", len(records))
    return records


@sales_mcp_server.tool()
def get_promo_period():
    """Return promotion start and end date based on sales data."""
    logger.debug("get_promo_period invoked")
    config = _get_config()
    df = load_sales(config)
    promo_df = df[df["is_promotion"] == True]
    promo_start = promo_df["transaction_date"].min()
    promo_end = promo_df["transaction_date"].max()
    result = {
        "promo_start": str(promo_start.date()),
        "promo_end": str(promo_end.date()),
    }
    logger.debug("get_promo_period returning %s", result)
    return result


@sales_mcp_server.tool()
def get_promo_sales_by_store():
    """Return total promotion-period sales by store."""
    logger.debug("get_promo_sales_by_store invoked")
    config = _get_config()
    df = load_sales(config)
    promo_df = df[df["is_promotion"] == True]
    promo_sales = (
        promo_df.groupby(["store_id", "store_name"], as_index=False)["quantity_sold"]
        .sum()
        .rename(columns={"quantity_sold": "promo_qty_sold"})
    )
    records = promo_sales.to_dict(orient="records")
    logger.debug("get_promo_sales_by_store returning %s records", len(records))
    return records


@sales_mcp_server.tool()
def get_sales_data():
    """Return sales data as list of dicts."""
    logger.debug("get_sales_data invoked")
    config = _get_config()
    df = load_sales(config)
    records = df.to_dict(orient="records")
    logger.debug("get_sales_data returning %s records", len(records))
    return records


def init_sales_mcp_server(config: AppConfig) -> FastMCP:
    global _sales_config
    _sales_config = config
    logger.info("Sales MCP server initialized")
    return sales_mcp_server
