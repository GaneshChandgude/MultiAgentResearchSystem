from __future__ import annotations

import asyncio
import importlib
import logging
import importlib.util
import inspect
import threading
from typing import Any, Dict, Iterable, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model

from .toolset_registry import Toolset

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


def _normalize_sse_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/sse"):
        return url
    return f"{url}/sse"


class _AsyncioThreadRunner:
    """Execute coroutines on a dedicated event loop thread."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mcp-toolset-loop", daemon=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    def run(self, coro: Any) -> Any:
        if self._closed or self._loop is None:
            raise RuntimeError("Failed to initialize dedicated asyncio loop for MCP calls.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1.0)
        self._closed = True


_runner_lock = threading.Lock()
_runner: _AsyncioThreadRunner | None = None


def _get_runner() -> _AsyncioThreadRunner:
    global _runner
    if _runner is not None:
        return _runner
    with _runner_lock:
        if _runner is None:
            _runner = _AsyncioThreadRunner()
    return _runner


def _run_coro(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    return _get_runner().run(coro)


def shutdown_mcp_runtime() -> None:
    global _runner
    with _runner_lock:
        if _runner is None:
            return
        _runner.close()
        _runner = None


def _load_mcp_client() -> tuple[Any, Any]:
    if importlib.util.find_spec("mcp") is None:
        raise ModuleNotFoundError(
            "Missing 'mcp' dependency. Install it with `pip install mcp` "
            "or add it to your environment before using MCP toolsets."
        )
    client_module = importlib.import_module("mcp.client")
    client_session = getattr(client_module, "ClientSession", None)

    if client_session is None and importlib.util.find_spec("mcp.client.session") is not None:
        session_module = importlib.import_module("mcp.client.session")
        client_session = getattr(session_module, "ClientSession", None)

    if client_session is None:
        raise ImportError(
            "Unable to locate ClientSession in the installed 'mcp' package. "
            "Please verify the MCP client package version."
        )

    if importlib.util.find_spec("mcp.client.sse") is None:
        raise ImportError(
            "Unable to locate the MCP SSE client in the installed 'mcp' package. "
            "Please verify the MCP client package version."
        )
    sse_module = importlib.import_module("mcp.client.sse")
    sse_client = getattr(sse_module, "sse_client", None)
    if sse_client is None:
        raise ImportError(
            "Unable to locate sse_client in the installed 'mcp' package. "
            "Please verify the MCP client package version."
        )

    return client_session, sse_client


def _tool_field(tool_info: Any, field: str, fallback: str | None = None) -> Any:
    if isinstance(tool_info, dict):
        return tool_info.get(field) or (tool_info.get(fallback) if fallback else None)
    return getattr(tool_info, field, None) or (getattr(tool_info, fallback, None) if fallback else None)


class MCPToolsetClient:
    def __init__(self, base_url: str, headers: Dict[str, str] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.sse_url = _normalize_sse_url(base_url)
        self.headers = {str(key): str(value) for key, value in (headers or {}).items() if key and value}

    def list_tools(self) -> List[Any]:
        return _run_coro(self._list_tools())

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        return _run_coro(self._call_tool(tool_name, arguments))

    async def _list_tools(self) -> List[Any]:
        ClientSession, sse_client = _load_mcp_client()

        sse_kwargs: Dict[str, Any] = {}
        if self.headers and "headers" in inspect.signature(sse_client).parameters:
            sse_kwargs["headers"] = self.headers

        async with sse_client(self.sse_url, **sse_kwargs) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()

        if isinstance(result, dict):
            return result.get("tools", [])
        return getattr(result, "tools", result)

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        ClientSession, sse_client = _load_mcp_client()

        sse_kwargs: Dict[str, Any] = {}
        if self.headers and "headers" in inspect.signature(sse_client).parameters:
            sse_kwargs["headers"] = self.headers

        async with sse_client(self.sse_url, **sse_kwargs) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

        if isinstance(result, dict) and "content" in result:
            return result["content"]
        return result


def _build_args_schema(tool_name: str, input_schema: Dict[str, Any]) -> type[BaseModel] | None:
    if not input_schema:
        return None
    properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    if not properties:
        return None
    required = set(input_schema.get("required", [])) if isinstance(input_schema, dict) else set()
    fields = {}
    for prop in properties:
        default = ... if prop in required else None
        fields[prop] = (Any, default)
    return create_model(f"{tool_name}Args", **fields)


def _build_tool(client: MCPToolsetClient, tool_info: Any) -> StructuredTool:
    tool_name = _tool_field(tool_info, "name") or "unknown"
    description = _tool_field(tool_info, "description") or ""
    input_schema = _tool_field(tool_info, "inputSchema", "input_schema") or {}
    args_schema = _build_args_schema(tool_name, input_schema)

    def handler(**kwargs):
        return client.call_tool(tool_name, kwargs)

    handler.__name__ = tool_name
    return StructuredTool.from_function(
        func=handler,
        name=tool_name,
        description=description,
        args_schema=args_schema,
    )


def build_mcp_toolset(name: str, description: str, base_url: str, headers: Dict[str, str] | None = None) -> Toolset:
    client = MCPToolsetClient(base_url, headers=headers)
    tools = []
    for tool_info in client.list_tools():
        tool_name = _tool_field(tool_info, "name") or "unknown"
        logger.debug("Registering MCP tool %s from %s", tool_name, base_url)
        tools.append(_build_tool(client, tool_info))
    return Toolset(name=name, tools=tools, description=description)
