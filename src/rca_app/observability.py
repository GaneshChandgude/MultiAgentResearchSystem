from __future__ import annotations

import importlib.util
import inspect
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from .config import AppConfig

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


def _langfuse_verify_setting(config: AppConfig) -> bool | str:
    if config.langfuse_ca_bundle:
        return config.langfuse_ca_bundle
    return config.langfuse_verify_ssl


def _build_langfuse_httpx_client(config: AppConfig) -> Any | None:
    verify_setting = _langfuse_verify_setting(config)
    if verify_setting is True:
        return None
    try:
        import httpx
    except ModuleNotFoundError:
        logger.warning(
            "Langfuse configured with SSL verification overrides but httpx is missing."
        )
        return None
    return httpx.Client(verify=verify_setting)


def build_langfuse_callbacks(
    config: AppConfig,
    user_id: str | None = None,
    query_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    tags: List[str] | None = None,
    trace_context: Dict[str, Any] | None = None,
) -> List[BaseCallbackHandler]:
    if not config.langfuse_enabled:
        return []
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.warning("Langfuse enabled but keys are missing; disabling callbacks.")
        return []
    if not importlib.util.find_spec("langfuse"):
        logger.warning("Langfuse enabled but package is not installed; disabling callbacks.")
        return []

    try:
        from langfuse.langchain import CallbackHandler
    except ModuleNotFoundError:
        logger.warning(
            "Langfuse enabled but callback module is unavailable; disabling callbacks."
        )
        return []

    handler_kwargs: Dict[str, Any] = {
        "public_key": config.langfuse_public_key,
        "secret_key": config.langfuse_secret_key,
        "host": config.langfuse_host,
        "debug": config.langfuse_debug,
    }
    httpx_client = _build_langfuse_httpx_client(config)
    if httpx_client is not None:
        handler_kwargs["httpx_client"] = httpx_client
    if config.langfuse_release:
        handler_kwargs["release"] = config.langfuse_release
    if user_id:
        handler_kwargs["user_id"] = user_id
    if query_id:
        handler_kwargs["session_id"] = query_id
    if metadata:
        handler_kwargs["metadata"] = metadata
    if tags:
        handler_kwargs["tags"] = tags
    if trace_context:
        handler_kwargs["trace_context"] = trace_context

    supported_params = set(inspect.signature(CallbackHandler).parameters)
    if "update_trace" in supported_params:
        update_trace: Dict[str, Any] = {}
        if user_id:
            update_trace["user_id"] = user_id
        if query_id:
            update_trace["session_id"] = query_id
        if metadata:
            update_trace["metadata"] = metadata
        if tags:
            update_trace["tags"] = tags
        if update_trace:
            handler_kwargs["update_trace"] = update_trace
    filtered_kwargs = {key: value for key, value in handler_kwargs.items() if key in supported_params}
    if filtered_kwargs.keys() != handler_kwargs.keys():
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.langfuse_public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", config.langfuse_secret_key)
        os.environ.setdefault("LANGFUSE_HOST", config.langfuse_host)
        os.environ.setdefault("LANGFUSE_RELEASE", config.langfuse_release)
        os.environ.setdefault("LANGFUSE_DEBUG", str(config.langfuse_debug).lower())
        logger.debug(
            "Langfuse callback args filtered to %s based on installed handler signature.",
            ", ".join(sorted(filtered_kwargs)) or "<none>",
        )
    return [CallbackHandler(**filtered_kwargs)]


def supports_langfuse_trace_context(config: AppConfig) -> bool:
    if not config.langfuse_enabled:
        return False
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        return False
    if not importlib.util.find_spec("langfuse"):
        return False
    try:
        from langfuse.langchain import CallbackHandler
    except ModuleNotFoundError:
        return False
    return "trace_context" in inspect.signature(CallbackHandler).parameters


def build_langfuse_client(config: AppConfig):
    if not config.langfuse_enabled:
        return None
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.warning("Langfuse enabled but keys are missing; skipping client.")
        return None
    if not importlib.util.find_spec("langfuse"):
        logger.warning("Langfuse enabled but package is not installed; skipping client.")
        return None

    from langfuse import Langfuse

    client_kwargs: Dict[str, Any] = {
        "public_key": config.langfuse_public_key,
        "secret_key": config.langfuse_secret_key,
        "host": config.langfuse_host,
        "debug": config.langfuse_debug,
    }
    httpx_client = _build_langfuse_httpx_client(config)
    if httpx_client is not None:
        client_kwargs["httpx_client"] = httpx_client
    if config.langfuse_release:
        client_kwargs["release"] = config.langfuse_release
    return Langfuse(**client_kwargs)


def build_langfuse_invoke_config(
    config: AppConfig,
    user_id: str | None = None,
    query_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    tags: List[str] | None = None,
    trace_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    enriched_metadata: Dict[str, Any] = dict(metadata or {})
    if user_id:
        enriched_metadata.setdefault("langfuse_user_id", user_id)
    if query_id:
        enriched_metadata.setdefault("langfuse_session_id", query_id)
    callbacks = build_langfuse_callbacks(
        config,
        user_id=user_id,
        query_id=query_id,
        metadata=enriched_metadata,
        tags=tags,
        trace_context=trace_context,
    )
    if not callbacks:
        return {}
    return {"callbacks": callbacks, "tags": tags or [], "metadata": enriched_metadata}
