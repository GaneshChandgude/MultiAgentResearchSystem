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
    )
