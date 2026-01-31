from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd

from .config import AppConfig

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


def sales_path(config: AppConfig) -> Path:
    return config.data_dir / "sales_transactions.csv"


def inventory_path(config: AppConfig) -> Path:
    return config.data_dir / "inventory_transactions.csv"


def load_sales(config: AppConfig) -> pd.DataFrame:
    path = sales_path(config)
    logger.debug("Loading sales data from %s", path)
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    logger.debug("Loaded sales data rows=%s columns=%s", len(df), list(df.columns))
    return df


def load_inventory(config: AppConfig) -> pd.DataFrame:
    path = inventory_path(config)
    logger.debug("Loading inventory data from %s", path)
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    logger.debug("Loaded inventory data rows=%s columns=%s", len(df), list(df.columns))
    return df
