from __future__ import annotations

import logging
from typing import Any, Dict, List, NotRequired, TypedDict

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


class RCAState(TypedDict):
    task: str
    output: str
    trace: List[Dict[str, Any]]
    history: NotRequired[List[Any]]
    todos: NotRequired[List[Dict[str, Any]]]
    todo_progress: NotRequired[Dict[str, Any]]


logger.debug("RCAState typing loaded")
