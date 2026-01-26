from __future__ import annotations

from langchain.tools import tool

from .config import AppConfig
from .data import load_sales


def build_sales_tools(config: AppConfig):
    @tool
    def get_daily_sales():
        """Return daily aggregated sales by store."""
        df = load_sales(config)
        daily = (
            df.groupby(["transaction_date", "store_id", "store_name"], as_index=False)[
                "quantity_sold"
            ]
            .sum()
            .sort_values(["transaction_date", "store_id"])
        )
        return daily.to_dict(orient="records")

    @tool
    def get_promo_period():
        """Return promotion start and end date based on sales data."""
        df = load_sales(config)
        promo_df = df[df["is_promotion"] == True]
        promo_start = promo_df["transaction_date"].min()
        promo_end = promo_df["transaction_date"].max()
        return {
            "promo_start": str(promo_start.date()),
            "promo_end": str(promo_end.date()),
        }

    @tool
    def get_promo_sales_by_store():
        """Return total promotion-period sales by store."""
        df = load_sales(config)
        promo_df = df[df["is_promotion"] == True]
        promo_sales = (
            promo_df.groupby(["store_id", "store_name"], as_index=False)["quantity_sold"]
            .sum()
            .rename(columns={"quantity_sold": "promo_qty_sold"})
        )
        return promo_sales.to_dict(orient="records")

    @tool
    def get_sales_data():
        """Return sales data as list of dicts."""
        df = load_sales(config)
        return df.to_dict(orient="records")

    return [get_daily_sales, get_promo_period, get_promo_sales_by_store, get_sales_data]
