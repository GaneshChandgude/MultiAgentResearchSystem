from __future__ import annotations

import logging
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from .config import AppConfig
from .observability import build_langfuse_callbacks

logger = logging.getLogger(__name__)


def get_llm_model(config: AppConfig) -> AzureChatOpenAI:
    if not (config.azure_openai_endpoint and config.azure_openai_api_key and config.azure_openai_deployment):
        raise ValueError("Azure OpenAI endpoint, api key, and deployment must be set.")

    logger.debug(
        "Initializing AzureChatOpenAI deployment=%s endpoint=%s api_version=%s",
        config.azure_openai_deployment,
        config.azure_openai_endpoint,
        config.azure_openai_api_version,
    )
    callbacks = build_langfuse_callbacks(config)
    if callbacks:
        logger.info("Langfuse enabled for LLM observability")

    return AzureChatOpenAI(
        azure_endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.azure_openai_api_version,
        model=config.azure_openai_deployment,
        azure_deployment=config.azure_openai_deployment,
        temperature=0.7,
        timeout=300,
        max_retries=3,
        callbacks=callbacks or None,
    )


def get_embeddings(config: AppConfig) -> AzureOpenAIEmbeddings:
    if not (config.embeddings_endpoint and config.embeddings_api_key):
        raise ValueError("Azure OpenAI embeddings endpoint and api key must be set.")

    logger.debug(
        "Initializing AzureOpenAIEmbeddings model=%s endpoint=%s api_version=%s",
        config.embeddings_model,
        config.embeddings_endpoint,
        config.embeddings_api_version,
    )
    return AzureOpenAIEmbeddings(
        model=config.embeddings_model,
        azure_endpoint=config.embeddings_endpoint,
        api_key=config.embeddings_api_key,
        openai_api_version=config.embeddings_api_version,
    )
