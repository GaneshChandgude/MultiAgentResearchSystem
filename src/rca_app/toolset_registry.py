from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


@dataclass
class Toolset:
    name: str
    tools: List[Any]
    description: str
    _tool_lookup: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._rebuild_lookup()

    def _rebuild_lookup(self) -> None:
        self._tool_lookup = {}
        for tool in self.tools:
            tool_name = getattr(tool, "name", None)
            if tool_name:
                self._tool_lookup[tool_name] = tool

    def register_tool(self, tool: Any) -> None:
        tool_name = getattr(tool, "name", None)
        if not tool_name:
            raise ValueError("Tool must have a name to be registered.")
        if tool_name in self._tool_lookup:
            logger.warning("Overwriting tool %s in toolset %s", tool_name, self.name)
        self.tools.append(tool)
        self._tool_lookup[tool_name] = tool

    def get_tool(self, tool_name: str) -> Any:
        if tool_name in self._tool_lookup:
            return self._tool_lookup[tool_name]
        raise KeyError(f"Tool '{tool_name}' not found in toolset '{self.name}'.")


class ToolsetRegistry:
    def __init__(self, toolsets: Iterable[Toolset] | None = None) -> None:
        self._toolsets: Dict[str, Toolset] = {}
        self._tools: Dict[str, Any] = {}
        if toolsets:
            for toolset in toolsets:
                self.register(toolset)

    def register(self, toolset: Toolset) -> None:
        if toolset.name in self._toolsets:
            logger.warning("Overwriting toolset %s", toolset.name)
        self._toolsets[toolset.name] = toolset
        for tool_name, tool in toolset._tool_lookup.items():
            if tool_name in self._tools:
                logger.warning(
                    "Tool name collision for %s (replaced by toolset %s)",
                    tool_name,
                    toolset.name,
                )
            self._tools[tool_name] = tool

    def get_toolset(self, name: str) -> Toolset:
        if name in self._toolsets:
            return self._toolsets[name]
        raise KeyError(f"Toolset '{name}' not found.")

    def find_tool(self, tool_name: str) -> Any:
        if tool_name in self._tools:
            return self._tools[tool_name]
        raise KeyError(f"Tool '{tool_name}' not found in registered toolsets.")

    def all_tools(self) -> List[Any]:
        return list(self._tools.values())
