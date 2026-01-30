from __future__ import annotations

import importlib.util
import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from .config import AppConfig

logger = logging.getLogger(__name__)


def build_langfuse_callbacks(
    config: AppConfig,
    user_id: str | None = None,
    query_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    tags: List[str] | None = None,
) -> List[BaseCallbackHandler]:
    if not config.langfuse_enabled:
        return []
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        logger.warning("Langfuse enabled but keys are missing; disabling callbacks.")
        return []
    if not importlib.util.find_spec("langfuse"):
        logger.warning("Langfuse enabled but package is not installed; disabling callbacks.")
        return []

    from langfuse.callback import CallbackHandler

    handler_kwargs: Dict[str, Any] = {
        "public_key": config.langfuse_public_key,
        "secret_key": config.langfuse_secret_key,
        "host": config.langfuse_host,
        "debug": config.langfuse_debug,
    }
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

    return [CallbackHandler(**handler_kwargs)]


def build_langfuse_invoke_config(
    config: AppConfig,
    user_id: str | None = None,
    query_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
    tags: List[str] | None = None,
) -> Dict[str, Any]:
    callbacks = build_langfuse_callbacks(
        config,
        user_id=user_id,
        query_id=query_id,
        metadata=metadata,
        tags=tags,
    )
    if not callbacks:
        return {}
    return {"callbacks": callbacks, "tags": tags or [], "metadata": metadata or {}}
