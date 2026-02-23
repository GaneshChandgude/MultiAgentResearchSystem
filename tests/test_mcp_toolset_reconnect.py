import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import types

# Minimal stubs so mcp_toolset can be imported without optional langchain dependency.
langchain_core = types.ModuleType("langchain_core")
langchain_core_tools = types.ModuleType("langchain_core.tools")


class _StructuredTool:
    @classmethod
    def from_function(cls, **_kwargs):
        return object()


langchain_core_tools.StructuredTool = _StructuredTool
sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.tools", langchain_core_tools)

pydantic = types.ModuleType("pydantic")


class _BaseModel:
    pass


def _create_model(_name, **_fields):
    return _BaseModel


pydantic.BaseModel = _BaseModel
pydantic.create_model = _create_model
sys.modules.setdefault("pydantic", pydantic)

from rca_app import mcp_toolset
from rca_app.mcp_toolset import MCPToolsetClient



def _cleanup_client(client: MCPToolsetClient) -> None:
    client._is_closed = True
    with mcp_toolset._clients_lock:
        mcp_toolset._clients.discard(client)


def test_invoke_with_reconnect_does_not_retry_non_recoverable_errors():
    client = MCPToolsetClient("http://localhost:9999")
    calls = {"count": 0}

    async def fake_get_session():
        return object()

    async def always_fail(_session):
        calls["count"] += 1
        raise ValueError("invalid arguments")

    client._get_session = fake_get_session  # type: ignore[assignment]

    with pytest.raises(ValueError, match="invalid arguments"):
        asyncio.run(client._invoke_with_reconnect("call_tool", "get_me", always_fail))

    assert calls["count"] == 1
    _cleanup_client(client)


def test_invoke_with_reconnect_recovers_when_close_fails(monkeypatch):
    client = MCPToolsetClient("http://localhost:9999")
    calls = {"count": 0, "connect": 0}

    async def fake_get_session():
        return object()

    async def fail_then_succeed(_session):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("Already connected to a transport")
        return {"ok": True}

    async def broken_close():
        raise RuntimeError("close failed")

    async def fake_connect():
        calls["connect"] += 1
        return object()

    async def fake_sleep(_seconds):
        return None

    client._get_session = fake_get_session  # type: ignore[assignment]
    client._close_session = broken_close  # type: ignore[assignment]
    client._connect_session = fake_connect  # type: ignore[assignment]
    monkeypatch.setattr(mcp_toolset.asyncio, "sleep", fake_sleep)

    result = asyncio.run(client._invoke_with_reconnect("call_tool", "get_me", fail_then_succeed))

    assert result == {"ok": True}
    assert calls["count"] == 2
    assert calls["connect"] == 1
    _cleanup_client(client)
