from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import AppConfig


def sales_path(config: AppConfig) -> Path:
    return config.data_dir / "sales_transactions.csv"


def inventory_path(config: AppConfig) -> Path:
    return config.data_dir / "inventory_transactions.csv"


def load_sales(config: AppConfig) -> pd.DataFrame:
    return pd.read_csv(sales_path(config), parse_dates=["transaction_date"])


def load_inventory(config: AppConfig) -> pd.DataFrame:
    return pd.read_csv(inventory_path(config), parse_dates=["transaction_date"])
