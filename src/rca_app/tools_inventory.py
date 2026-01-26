from __future__ import annotations

from langchain.tools import tool
import pandas as pd

from .config import AppConfig
from .data import load_inventory, load_sales


def build_inventory_tools(config: AppConfig):
    @tool
    def get_unique_stores() -> dict:
        """Return list of unique store_ids from sales and inventory timeline."""
        sales_df = load_sales(config)
        inv_df = load_inventory(config)
        stores = sorted(pd.concat([sales_df["store_id"], inv_df["store_id"]]).dropna().unique())
        return {"stores": stores}

    def drop_store_name(df):
        return df.drop(columns=["store_name"], errors="ignore")

    @tool
    def theoretical_onhand_after_promo_sales(promo_start: str, promo_end: str):
        """
        Compute theoretical on-hand inventory after promo sales for each store.
        Inputs: promo_start (YYYY-MM-DD), promo_end (YYYY-MM-DD)
        """
        sales = load_sales(config)
        inv = load_inventory(config)
        promo_start_dt = pd.to_datetime(promo_start)
        promo_end_dt = pd.to_datetime(promo_end)

        as_of = promo_start_dt - pd.Timedelta(days=1)

        inv_receipts_before = inv[
            (inv["transaction_date"] <= as_of)
            & (inv["transaction_type"] == "RECEIPT")
        ]

        start_inv = (
            inv_receipts_before.groupby("destination_location", as_index=False)["quantity"]
            .sum()
            .rename(
                columns={
                    "destination_location": "store_id",
                    "quantity": "start_receipt_qty",
                }
            )
        )

        inv_changes = inv[inv["transaction_date"] > as_of].copy()
        inv_changes["destination_location"] = inv_changes["destination_location"].replace(
            "NONE", pd.NA
        )
        inv_changes["target_store"] = inv_changes["destination_location"].fillna(
            inv_changes["store_id"]
        )

        inv_net_after = (
            inv_changes.groupby("target_store", as_index=False)["quantity"]
            .sum()
            .rename(columns={"target_store": "store_id", "quantity": "net_qty_after"})
        )

        promo_repl = inv[
            (inv["transaction_date"] == promo_start_dt)
            & (inv["transaction_type"] == "RECEIPT")
        ]

        promo_repl_by_store = (
            promo_repl.groupby("destination_location", as_index=False)["quantity"]
            .sum()
            .rename(
                columns={"destination_location": "store_id", "quantity": "promo_repl_qty"}
            )
        )

        promo_sales = sales[
            (sales["transaction_date"] >= promo_start_dt)
            & (sales["transaction_date"] <= promo_end_dt)
        ]

        promo_by_store = (
            promo_sales.groupby(["store_id", "store_name"], as_index=False)["quantity_sold"]
            .sum()
            .rename(columns={"quantity_sold": "promo_qty_sold"})
        )

        stores = sales[["store_id", "store_name"]].drop_duplicates()

        summary = (
            stores.merge(drop_store_name(start_inv), on="store_id", how="left")
            .merge(drop_store_name(inv_net_after), on="store_id", how="left")
            .merge(drop_store_name(promo_by_store), on="store_id", how="left")
            .merge(drop_store_name(promo_repl_by_store), on="store_id", how="left")
        ).fillna(0)

        summary["theoretical_after_changes"] = (
            summary["start_receipt_qty"] + summary["net_qty_after"]
        )

        summary["theoretical_onhand_after_promo_sales"] = (
            summary["theoretical_after_changes"] - summary["promo_qty_sold"]
        )

        return summary.to_dict(orient="records")

    @tool
    def get_daily_inventory_for_store(store_id: str):
        """
        Return daily inventory on-hand timeline for a given store_id.
        Computes daily movements, sales, net change, running inventory.
        """
        inv = load_inventory(config)
        sales = load_sales(config)

        inv["destination_location"] = inv["destination_location"].replace("NONE", pd.NA)
        inv["store"] = inv["store_id"]

        daily_inv_moves = (
            inv.groupby(["transaction_date", "store"], as_index=False)["quantity"].sum()
        )

        daily_sales = (
            sales.groupby(["transaction_date", "store_id"], as_index=False)["quantity_sold"]
            .sum()
            .rename(columns={"store_id": "store"})
        )

        timeline = pd.merge(
            daily_inv_moves, daily_sales, on=["transaction_date", "store"], how="outer"
        ).fillna(0)

        timeline["net_change"] = timeline["quantity"] - timeline["quantity_sold"]
        timeline = timeline.sort_values(["store", "transaction_date"])
        timeline["running_inventory"] = timeline.groupby("store")["net_change"].cumsum()

        result = timeline[timeline["store"] == store_id]
        return result.to_dict(orient="records")

    @tool
    def get_adjustments():
        """Return all shrinkage/adjustment rows."""
        df = load_inventory(config)
        adjustments = df[df["transaction_type"] == "ADJUSTMENT"]
        return adjustments.to_dict(orient="records")

    @tool
    def get_shrinkage_before_promo(promo_start: str):
        """Return shrinkage rows before promo start."""
        df = load_inventory(config)
        promo_start_dt = pd.to_datetime(promo_start)
        result = df[
            (df["transaction_type"] == "ADJUSTMENT")
            & (df["transaction_date"] < promo_start_dt)
        ]
        return result.to_dict(orient="records")

    @tool
    def get_shrinkage_during_promo(promo_start: str, promo_end: str):
        """Return shrinkage during promo."""
        df = load_inventory(config)
        promo_start_dt = pd.to_datetime(promo_start)
        promo_end_dt = pd.to_datetime(promo_end)
        result = df[
            (df["transaction_type"] == "ADJUSTMENT")
            & (df["transaction_date"] >= promo_start_dt)
            & (df["transaction_date"] <= promo_end_dt)
        ]
        return result.to_dict(orient="records")

    @tool
    def get_delayed_replenishments():
        """Return all inventory rows with DELAYED note."""
        df = load_inventory(config)
        delayed = df[df["notes"].str.contains("DELAYED", na=False)]
        return delayed.to_dict(orient="records")

    @tool
    def get_promo_replenishment_for_date(date: str):
        """Return receipts for given date."""
        df = load_inventory(config)
        date_dt = pd.to_datetime(date)
        promo_repl = df[
            (df["transaction_date"] == date_dt)
            & (df["transaction_type"] == "RECEIPT")
        ]
        return promo_repl.to_dict(orient="records")

    @tool
    def get_all_transfers():
        """Return all transfer rows."""
        df = load_inventory(config)
        transfers = df[df["transaction_type"] == "TRANSFER"]
        return transfers.to_dict(orient="records")

    @tool
    def get_transfers_for_date(date: str):
        """Return transfers for a given date."""
        df = load_inventory(config)
        date_dt = pd.to_datetime(date)
        result = df[
            (df["transaction_type"] == "TRANSFER")
            & (df["transaction_date"] == date_dt)
        ]
        return result.to_dict(orient="records")

    @tool
    def get_emergency_receipts():
        """Return emergency receipts."""
        df = load_inventory(config)
        emergency = df[df["notes"].str.contains("Emergency", na=False)]
        return emergency.to_dict(orient="records")

    @tool
    def get_inventory_data():
        """Return inventory movements as list of dicts."""
        df = load_inventory(config)
        return df.to_dict(orient="records")

    return [
        get_unique_stores,
        theoretical_onhand_after_promo_sales,
        get_daily_inventory_for_store,
        get_adjustments,
        get_shrinkage_before_promo,
        get_shrinkage_during_promo,
        get_delayed_replenishments,
        get_promo_replenishment_for_date,
        get_all_transfers,
        get_transfers_for_date,
        get_emergency_receipts,
        get_inventory_data,
    ]
