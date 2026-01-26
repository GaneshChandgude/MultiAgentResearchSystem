from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class RCAState(TypedDict):
    task: str
    output: str
    trace: List[Dict[str, Any]]
    history: List[Any]
