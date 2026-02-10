from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)


@dataclass(frozen=True)
class AppConfig:
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str
    planning_azure_openai_deployment: str
    specialist_azure_openai_deployment: str
    azure_openai_api_version: str
    embeddings_model: str
    embeddings_endpoint: str
    embeddings_api_key: str
    embeddings_api_version: str
    data_dir: Path
    salesforce_mcp_url: str
    sap_mcp_url: str
    langfuse_enabled: bool
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str
    langfuse_release: str
    langfuse_debug: bool
    langfuse_prompt_enabled: bool
    langfuse_prompt_label: str
    langfuse_verify_ssl: bool
    langfuse_ca_bundle: str
    pii_middleware_enabled: bool
    pii_redaction_enabled: bool
    pii_block_input: bool
    nested_agent_pii_profile: Literal["full", "nested", "off"]
    orchestrator_agent_pii_profile: Literal["full", "nested", "off"]
    max_input_length: int
    max_output_length: int
    model_guardrails_enabled: bool
    model_guardrails_moderation_enabled: bool
    model_guardrails_output_language: str
    recursion_limit: int


DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
DEFAULT_EMBEDDINGS_API_VERSION = "2023-05-15"
DEFAULT_EMBEDDINGS_MODEL = "TxtEmbedAda002"


def resolve_data_dir() -> Path:
    env_dir = os.getenv("RCA_DATA_DIR")
    if env_dir:
        logger.info("Using RCA data directory from RCA_DATA_DIR")
        return Path(env_dir).expanduser().resolve()
    logger.info("Using default RCA data directory")
    return Path(__file__).resolve().parents[2] / "data"


def load_config() -> AppConfig:
    logger.info("Loading RCA configuration")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    planning_deployment = os.getenv("AZURE_OPENAI_PLANNING_DEPLOYMENT", "").strip() or deployment
    specialist_deployment = os.getenv("AZURE_OPENAI_SPECIALIST_DEPLOYMENT", "").strip() or deployment

    embeddings_endpoint = os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT", endpoint).strip()
    embeddings_api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY", api_key).strip()
    salesforce_mcp_url = os.getenv("RCA_MCP_SALESFORCE_URL", "http://localhost:8600").strip()
    sap_mcp_url = os.getenv("RCA_MCP_SAP_URL", "http://localhost:8700").strip()
    langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
    langfuse_release = os.getenv("LANGFUSE_RELEASE", "").strip()
    langfuse_debug = os.getenv("LANGFUSE_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
    langfuse_prompt_enabled = os.getenv("LANGFUSE_PROMPT_ENABLED", "").strip().lower()
    if langfuse_prompt_enabled:
        prompt_enabled = langfuse_prompt_enabled in {"1", "true", "yes", "on"}
    else:
        prompt_enabled = langfuse_enabled
    langfuse_prompt_label = os.getenv("LANGFUSE_PROMPT_LABEL", "production").strip()
    langfuse_verify_ssl = os.getenv("LANGFUSE_VERIFY_SSL", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    langfuse_ca_bundle = os.getenv("LANGFUSE_CA_BUNDLE", "").strip()
    pii_redaction_enabled = os.getenv("RCA_PII_REDACTION_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    pii_middleware_enabled = os.getenv("RCA_PII_MIDDLEWARE_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    pii_block_input = os.getenv("RCA_PII_BLOCK_INPUT", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    nested_agent_pii_profile = os.getenv("RCA_NESTED_AGENT_PII_PROFILE", "nested").strip().lower() or "nested"
    if nested_agent_pii_profile not in {"full", "nested", "off"}:
        logger.warning(
            "Invalid RCA_NESTED_AGENT_PII_PROFILE=%s; falling back to 'nested'",
            nested_agent_pii_profile,
        )
        nested_agent_pii_profile = "nested"
    orchestrator_agent_pii_profile = os.getenv("RCA_ORCHESTRATOR_PII_PROFILE", "off").strip().lower() or "off"
    if orchestrator_agent_pii_profile not in {"full", "nested", "off"}:
        logger.warning(
            "Invalid RCA_ORCHESTRATOR_PII_PROFILE=%s; falling back to 'off'",
            orchestrator_agent_pii_profile,
        )
        orchestrator_agent_pii_profile = "off"
    max_input_length = int(os.getenv("RCA_MAX_INPUT_LENGTH", "4000").strip() or "4000")
    max_output_length = int(os.getenv("RCA_MAX_OUTPUT_LENGTH", "8000").strip() or "8000")
    model_guardrails_enabled = os.getenv("RCA_MODEL_GUARDRAILS_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    model_guardrails_moderation_enabled = os.getenv(
        "RCA_MODEL_GUARDRAILS_MODERATION_ENABLED",
        "true",
    ).strip().lower() in {"1", "true", "yes", "on"}
    model_guardrails_output_language = os.getenv("RCA_MODEL_GUARDRAILS_OUTPUT_LANGUAGE", "English").strip()
    recursion_limit = int(os.getenv("RCA_RECURSION_LIMIT", "50").strip() or "50")

    logger.debug(
        "Config resolved endpoint=%s deployment=%s data_dir=%s langfuse_enabled=%s",
        endpoint or "<unset>",
        deployment or "<unset>",
        resolve_data_dir(),
        langfuse_enabled,
    )
    return AppConfig(
        azure_openai_endpoint=endpoint,
        azure_openai_api_key=api_key,
        azure_openai_deployment=deployment,
        planning_azure_openai_deployment=planning_deployment,
        specialist_azure_openai_deployment=specialist_deployment,
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION),
        embeddings_model=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL),
        embeddings_endpoint=embeddings_endpoint,
        embeddings_api_key=embeddings_api_key,
        embeddings_api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION", DEFAULT_EMBEDDINGS_API_VERSION),
        data_dir=resolve_data_dir(),
        salesforce_mcp_url=salesforce_mcp_url,
        sap_mcp_url=sap_mcp_url,
        langfuse_enabled=langfuse_enabled,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=langfuse_host,
        langfuse_release=langfuse_release,
        langfuse_debug=langfuse_debug,
        langfuse_prompt_enabled=prompt_enabled,
        langfuse_prompt_label=langfuse_prompt_label,
        langfuse_verify_ssl=langfuse_verify_ssl,
        langfuse_ca_bundle=langfuse_ca_bundle,
        pii_middleware_enabled=pii_middleware_enabled,
        pii_redaction_enabled=pii_redaction_enabled,
        pii_block_input=pii_block_input,
        nested_agent_pii_profile=nested_agent_pii_profile,
        orchestrator_agent_pii_profile=orchestrator_agent_pii_profile,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        model_guardrails_enabled=model_guardrails_enabled,
        model_guardrails_moderation_enabled=model_guardrails_moderation_enabled,
        model_guardrails_output_language=model_guardrails_output_language,
        recursion_limit=recursion_limit,
    )
