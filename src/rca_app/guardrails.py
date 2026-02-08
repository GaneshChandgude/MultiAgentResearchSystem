from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List

from .config import AppConfig
logger = logging.getLogger(__name__)
logger.debug("Loaded module %s", __name__)

MAX_INPUT_LENGTH = 4000
MAX_OUTPUT_LENGTH = 8000

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

@dataclass(frozen=True)
class InputGuardrailResult:
    allowed: bool
    message: str
    sanitized: str


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


def apply_output_guardrails(response: str, *, config: AppConfig | None = None) -> str:
    redacted = response
    patterns = list(_SENSITIVE_OUTPUT_PATTERNS)
    if config is None or config.pii_redaction_enabled:
        patterns += _PII_PATTERNS

    for pattern in patterns:
        redacted = pattern.sub("[REDACTED]", redacted)

    max_output = config.max_output_length if config else MAX_OUTPUT_LENGTH
    if len(redacted) > max_output:
        logger.warning("Output exceeded max length; truncating.")
        redacted = f"{redacted[:max_output]}... [truncated]"

    return redacted


def apply_value_guardrails(value: Any, *, config: AppConfig | None = None) -> Any:
    if isinstance(value, str):
        return apply_output_guardrails(value, config=config)
    if isinstance(value, list):
        return [apply_value_guardrails(item, config=config) for item in value]
    if isinstance(value, dict):
        return {key: apply_value_guardrails(item, config=config) for key, item in value.items()}
    return value


def apply_tool_output_guardrails(
    trace: List[Dict[str, Any]], *, config: AppConfig | None = None
) -> List[Dict[str, Any]]:
    return [apply_value_guardrails(entry, config=config) for entry in trace]
