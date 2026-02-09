from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import resolve_data_dir

DEFAULT_LOG_FILE = "rca_app.log"
DEFAULT_LOG_LEVEL = "INFO"


def configure_logging() -> Path:
    log_path = os.getenv("RCA_LOG_FILE", "").strip()
    if log_path:
        log_file = Path(log_path).expanduser().resolve()
    else:
        log_file = resolve_data_dir() / DEFAULT_LOG_FILE

    log_level = os.getenv("RCA_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    log_to_console = os.getenv("RCA_LOG_TO_CONSOLE", "true").strip().lower()
    enable_console = log_to_console not in {"0", "false", "no", "off"}
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.FileHandler(log_file)]
    if enable_console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

    logging.getLogger(__name__).info(
        "Logging initialized at %s (level=%s)", log_file, log_level
    )
    return log_file
