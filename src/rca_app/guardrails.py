from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List

from langchain.agents.middleware import PIIMiddleware
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from .config import AppConfig
logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)

MAX_INPUT_LENGTH = 4000
MAX_OUTPUT_LENGTH = 8000
MODEL_GUARDRAIL_MAX_CHARS = 2000
MODEL_GUARDRAIL_BLOCK_MESSAGE = "Response blocked by content safety policy."

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"you\s+are\s+chatgpt", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"bypass\s+policy", re.IGNORECASE),
]

_SENSITIVE_REQUEST_PATTERNS = [
    re.compile(r"(api\s*key|secret|password|token)", re.IGNORECASE),
    re.compile(r"(show|reveal|expose|leak|print).*(api\s*key|secret|password|token)", re.IGNORECASE),
]

_SENSITIVE_OUTPUT_PATTERNS = [
    re.compile(r"(api\s*key|apikey|secret|password|token)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"sk-[A-Za-z0-9]{16,}", re.IGNORECASE),
]

_PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{2,4}\)|\d{2,4})[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
]

_PII_MIDDLEWARE_TYPES = ("email", "credit_card", "ip", "mac_address", "url")

@dataclass(frozen=True)
class InputGuardrailResult:
    allowed: bool
    message: str
    sanitized: str


@dataclass(frozen=True)
class ModelGuardrailResult:
    allowed: bool
    message: str
    categories: List[str]
    language: str


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip())


def apply_input_guardrails(query: str, *, config: AppConfig | None = None) -> InputGuardrailResult:
    sanitized = _normalize_query(query)
    if not sanitized:
        return InputGuardrailResult(False, "Query cannot be empty.", sanitized)

    max_input = config.max_input_length if config else MAX_INPUT_LENGTH
    if len(sanitized) > max_input:
        return InputGuardrailResult(
            False,
            f"Query is too long. Please keep it under {max_input} characters.",
            sanitized,
        )

    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(sanitized):
            logger.warning("Prompt injection pattern detected: %s", pattern.pattern)
            return InputGuardrailResult(
                False,
                "Query appears to include prompt injection or instruction-hijacking content.",
                sanitized,
            )

    for pattern in _SENSITIVE_REQUEST_PATTERNS:
        if pattern.search(sanitized):
            logger.warning("Sensitive data request detected: %s", pattern.pattern)
            return InputGuardrailResult(
                False,
                "Query appears to request sensitive information. Please rephrase without secrets requests.",
                sanitized,
            )

    if config and config.pii_block_input:
        for pattern in _PII_PATTERNS:
            if pattern.search(sanitized):
                logger.warning("PII pattern detected in input: %s", pattern.pattern)
                return InputGuardrailResult(
                    False,
                    "Query appears to contain personal data. Please remove PII before submitting.",
                    sanitized,
                )

    return InputGuardrailResult(True, "ok", sanitized)


def apply_output_guardrails(
    response: str,
    *,
    config: AppConfig | None = None,
    run_model_guardrails: bool = True,
) -> str:
    redacted = response
    patterns = list(_SENSITIVE_OUTPUT_PATTERNS)
    if config is None or config.pii_redaction_enabled:
        patterns += _PII_PATTERNS

    for pattern in patterns:
        redacted = pattern.sub("[REDACTED]", redacted)

    if run_model_guardrails and config and config.model_guardrails_enabled:
        model_result = apply_model_guardrails(redacted, config=config)
        if not model_result.allowed:
            return model_result.message

    max_output = config.max_output_length if config else MAX_OUTPUT_LENGTH
    if len(redacted) > max_output:
        logger.warning("Output exceeded max length; truncating.")
        redacted = f"{redacted[:max_output]}... [truncated]"

    return redacted


def apply_value_guardrails(value: Any, *, config: AppConfig | None = None) -> Any:
    if isinstance(value, str):
        return apply_output_guardrails(value, config=config, run_model_guardrails=False)
    if isinstance(value, list):
        return [apply_value_guardrails(item, config=config) for item in value]
    if isinstance(value, dict):
        return {key: apply_value_guardrails(item, config=config) for key, item in value.items()}
    return value


def apply_tool_output_guardrails(
    trace: List[Dict[str, Any]], *, config: AppConfig | None = None
) -> List[Dict[str, Any]]:
    return [apply_value_guardrails(entry, config=config) for entry in trace]


def apply_model_guardrails(response: str, *, config: AppConfig) -> ModelGuardrailResult:
    if not config.model_guardrails_enabled:
        return ModelGuardrailResult(True, "ok", [], "")

    if not (config.azure_openai_endpoint and config.azure_openai_api_key and config.azure_openai_deployment):
        logger.warning("Model guardrails enabled but Azure OpenAI credentials are missing.")
        return ModelGuardrailResult(True, "ok", [], "")

    moderation_enabled = config.model_guardrails_moderation_enabled
    required_language = (config.model_guardrails_output_language or "").strip()
    if not moderation_enabled and not required_language:
        return ModelGuardrailResult(True, "ok", [], "")

    model = _get_guardrail_model(config)
    content = response[:MODEL_GUARDRAIL_MAX_CHARS]

    import json

    system_prompt = (
        "You are a safety classifier. Return JSON only with keys: "
        '"allowed" (boolean), "categories" (array of strings), '
        '"language" (string), "reason" (string). '
        "If moderation_enabled is false, ignore content moderation checks. "
        "If required_language is set and the content is not primarily in that language, "
        'set allowed=false and reason="language_mismatch". '
        "If moderation is enabled and content includes unsafe material "
        "(toxicity, hate, harassment, sexual content, violence, self-harm, illegal content, or offensive language), "
        'set allowed=false and include the relevant categories.'
    )
    user_payload = {
        "moderation_enabled": moderation_enabled,
        "required_language": required_language,
        "content": content,
    }
    try:
        result = model.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=json.dumps(user_payload, ensure_ascii=False))]
        )
    except Exception:
        logger.exception("Model guardrails invocation failed.")
        return ModelGuardrailResult(True, "ok", [], "")

    parsed = _parse_guardrail_response(result.content)
    if not parsed.allowed:
        return ModelGuardrailResult(False, MODEL_GUARDRAIL_BLOCK_MESSAGE, parsed.categories, parsed.language)
    return parsed


def _get_guardrail_model(config: AppConfig) -> AzureChatOpenAI:
    cache_key = (
        config.azure_openai_endpoint,
        config.azure_openai_api_version,
        config.azure_openai_deployment,
        config.azure_openai_api_key,
    )
    if not hasattr(_get_guardrail_model, "_cache"):
        _get_guardrail_model._cache = {}
    cache = _get_guardrail_model._cache
    if cache_key in cache:
        return cache[cache_key]

    cache[cache_key] = AzureChatOpenAI(
        azure_endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.azure_openai_api_version,
        model=config.azure_openai_deployment,
        azure_deployment=config.azure_openai_deployment,
        temperature=0.0,
        timeout=60,
        max_retries=2,
    )
    return cache[cache_key]


def _parse_guardrail_response(raw: str) -> ModelGuardrailResult:
    import json

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse model guardrails response; allowing output.")
        return ModelGuardrailResult(True, "ok", [], "")

    allowed = bool(payload.get("allowed", True))
    categories = payload.get("categories") or []
    if not isinstance(categories, list):
        categories = [str(categories)]
    language = str(payload.get("language", "")).strip()
    message = str(payload.get("reason", "ok"))
    return ModelGuardrailResult(allowed, message, categories, language)


def build_pii_middleware(config: AppConfig) -> List[PIIMiddleware]:
    """Build LangChain PII middleware without any LLM-based detection."""
    if not config.pii_middleware_enabled:
        return []

    middleware: List[PIIMiddleware] = []

    if config.pii_block_input:
        middleware.extend(
            PIIMiddleware(
                pii_type,
                strategy="block",
                apply_to_input=True,
                apply_to_output=False,
                apply_to_tool_results=False,
            )
            for pii_type in _PII_MIDDLEWARE_TYPES
        )

    if config.pii_redaction_enabled:
        middleware.extend(
            PIIMiddleware(
                pii_type,
                strategy="redact",
                apply_to_input=False,
                apply_to_output=True,
                apply_to_tool_results=True,
            )
            for pii_type in _PII_MIDDLEWARE_TYPES
        )

    return middleware
