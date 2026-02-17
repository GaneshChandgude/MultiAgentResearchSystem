from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

import pandas as pd
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


def _serialize_result(result: Any, row_limit: int) -> dict[str, Any]:
    if isinstance(result, pd.DataFrame):
        output_df = result.head(row_limit)
        return {
            "result_type": "dataframe",
            "row_count": int(len(result)),
            "truncated": len(result) > len(output_df),
            "rows": output_df.to_dict(orient="records"),
        }

    if isinstance(result, pd.Series):
        output_series = result.head(row_limit)
        return {
            "result_type": "series",
            "row_count": int(len(result)),
            "truncated": len(result) > len(output_series),
            "rows": output_series.to_dict(),
        }

    if isinstance(result, (list, tuple)):
        rows = list(result)
        return {
            "result_type": "list",
            "row_count": len(rows),
            "truncated": len(rows) > row_limit,
            "rows": rows[:row_limit],
        }

    if isinstance(result, dict):
        return {
            "result_type": "dict",
            "row_count": len(result),
            "truncated": False,
            "rows": result,
        }

    return {
        "result_type": "scalar",
        "row_count": 1,
        "truncated": False,
        "rows": result,
    }


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


@sales_mcp_server.tool()
def execute_sales_sql(sql_query: str, row_limit: int = 200):
    """Execute a read-only SQL query against the sales table and return structured results.

    The query runs on an in-memory SQLite database with one table:
      - sales: all columns from sales_transactions.csv
    """
    logger.debug("execute_sales_sql invoked row_limit=%s", row_limit)
    cleaned_query = sql_query.strip().rstrip(";")
    if not re.match(r"^(select|with)\b", cleaned_query, flags=re.IGNORECASE):
        raise ValueError("Only read-only SELECT/CTE queries are allowed.")

    config = _get_config()
    df = load_sales(config)
    row_limit = max(1, min(int(row_limit), 1000))

    with sqlite3.connect(":memory:") as conn:
        df.to_sql("sales", conn, index=False, if_exists="replace")
        result_df = pd.read_sql_query(cleaned_query, conn)

    serialized = _serialize_result(result_df, row_limit=row_limit)
    response = {
        "query": cleaned_query,
        "table": "sales",
        "columns": list(df.columns),
        **serialized,
    }
    logger.debug("execute_sales_sql returning result_type=%s row_count=%s", response["result_type"], response["row_count"])
    return response


@sales_mcp_server.tool()
def execute_sales_dataframe_code(python_expression: str, row_limit: int = 200):
    """Execute a pandas expression against the sales DataFrame.

    Available variables:
      - df: pandas DataFrame for sales transactions
      - pd: pandas module

    Example expression:
      df[df['is_promotion'] == True].assign(revenue=df['quantity_sold'] * df['promotion_price'])
        .groupby(['product_id', 'product_name'], as_index=False)['revenue'].sum()
        .sort_values('revenue', ascending=False)
        .head(1)
    """
    logger.debug("execute_sales_dataframe_code invoked row_limit=%s", row_limit)
    config = _get_config()
    df = load_sales(config)
    row_limit = max(1, min(int(row_limit), 1000))

    safe_globals = {"__builtins__": {}}
    safe_locals = {"df": df.copy(), "pd": pd}
    result = eval(python_expression, safe_globals, safe_locals)

    serialized = _serialize_result(result, row_limit=row_limit)
    response = {
        "expression": python_expression,
        **serialized,
    }
    logger.debug(
        "execute_sales_dataframe_code returning result_type=%s row_count=%s",
        response["result_type"],
        response["row_count"],
    )
    return response


def init_sales_mcp_server(config: AppConfig) -> FastMCP:
    global _sales_config
    _sales_config = config
    logger.info("Sales MCP server initialized")
    return sales_mcp_server
