from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppConfig:
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str
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


DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"
DEFAULT_EMBEDDINGS_API_VERSION = "2023-05-15"
DEFAULT_EMBEDDINGS_MODEL = "TxtEmbedAda002"


def resolve_data_dir() -> Path:
    env_dir = os.getenv("RCA_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data"


def load_config() -> AppConfig:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

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

    return AppConfig(
        azure_openai_endpoint=endpoint,
        azure_openai_api_key=api_key,
        azure_openai_deployment=deployment,
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
    )
