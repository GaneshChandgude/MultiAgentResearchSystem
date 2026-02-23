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
    with mcp_toolset._client_cache_lock:
        stale_keys = [key for key, cached_client in mcp_toolset._client_cache.items() if cached_client is client]
        for key in stale_keys:
            mcp_toolset._client_cache.pop(key, None)
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


def test_call_tool_raises_when_mcp_result_is_error_payload():
    client = MCPToolsetClient("http://localhost:9999")

    async def fake_invoke_with_reconnect(**_kwargs):
        return {
            "isError": True,
            "content": [
                {
                    "type": "text",
                    "text": "parameter sort is not of type string, is <nil>",
                }
            ],
        }

    client._invoke_with_reconnect = fake_invoke_with_reconnect  # type: ignore[assignment]

    with pytest.raises(ValueError, match="parameter sort is not of type string"):
        asyncio.run(client._call_tool("list_repositories", {}))

    _cleanup_client(client)


def test_get_session_reconnects_when_cached_session_is_marked_closed():
    client = MCPToolsetClient("http://localhost:9999")

    class StaleSession:
        closed = True

    stale_session = StaleSession()
    fresh_session = object()
    calls = {"close": 0, "connect": 0}

    async def fake_close():
        calls["close"] += 1
        client._session = None

    async def fake_connect():
        calls["connect"] += 1
        client._session = fresh_session
        return fresh_session

    client._session = stale_session
    client._close_session = fake_close  # type: ignore[assignment]
    client._connect_session = fake_connect  # type: ignore[assignment]

    session = asyncio.run(client._get_session())

    assert session is fresh_session
    assert calls["close"] == 1
    assert calls["connect"] == 1
    _cleanup_client(client)


def test_get_mcp_client_reuses_same_instance_for_same_config():
    first = mcp_toolset.get_mcp_client("http://localhost:9999", headers={"Authorization": "Bearer abc"})
    second = mcp_toolset.get_mcp_client("http://localhost:9999/", headers={"Authorization": "Bearer abc"})

    assert first is second

    _cleanup_client(first)


def test_invoke_with_reconnect_closes_session_before_final_retryable_raise(monkeypatch):
    client = MCPToolsetClient("http://localhost:9999")
    calls = {"count": 0, "close": 0, "connect": 0}

    async def fake_get_session():
        return object()

    async def always_retryable_fail(_session):
        calls["count"] += 1
        raise RuntimeError("client disconnected")

    async def fake_close():
        calls["close"] += 1

    async def fake_connect():
        calls["connect"] += 1
        return object()

    async def fake_sleep(_seconds):
        return None

    client._get_session = fake_get_session  # type: ignore[assignment]
    client._close_session = fake_close  # type: ignore[assignment]
    client._connect_session = fake_connect  # type: ignore[assignment]

    monkeypatch.setattr(mcp_toolset.asyncio, "sleep", fake_sleep)
    with pytest.raises(RuntimeError, match="client disconnected"):
        asyncio.run(client._invoke_with_reconnect("call_tool", "get_me", always_retryable_fail))

    assert calls["count"] == 3
    assert calls["close"] == 3
    assert calls["connect"] == 2
    _cleanup_client(client)
