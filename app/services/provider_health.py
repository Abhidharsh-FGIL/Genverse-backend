"""LLM provider error classification and a per-process circuit breaker.

Retry policy (AI tailoring plan, pre-Phase-3 item 3):
  - retry only 429, 5xx, timeouts and connection errors;
  - 400 / 401 / 402 / 403 / 404 are non-retryable: go straight to the next
    provider (or the next model, for a 404 on one model).

Billing errors carried by another 4xx status (Anthropic's "credit balance is
too low" is an HTTP 400) are classified by error type/message as auth_billing.

Circuit breaker: after an auth/billing failure (401/402/403) the whole provider
is skipped for BREAKER_SECONDS. After a 404 (model not found) only that model is
skipped. Each trip is logged once at ERROR; skips while open log at DEBUG. State
is per process: each uvicorn/celery worker keeps its own breaker.
"""
import asyncio
import logging
import re
import time
from typing import Literal, Optional

logger = logging.getLogger(__name__)

BREAKER_SECONDS = 300.0

ErrorKind = Literal["retryable", "auth_billing", "not_found", "bad_request", "other"]

_open_until: dict[str, float] = {}
_LEADING_STATUS = re.compile(r"^\s*(?:Error code:\s*)?(\d{3})\b")


def error_status(exc: BaseException) -> Optional[int]:
    """HTTP status of a provider SDK error, if it has one."""
    for attr in ("status_code", "code"):           # openai/anthropic | google-genai
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    m = _LEADING_STATUS.match(str(exc))           # e.g. "402 RESOURCE_EXHAUSTED. ..."
    return int(m.group(1)) if m else None


# Billing failures that providers report with a non-billing status code.
# Anthropic: HTTP 400, error.type "invalid_request_error", message "Your credit
# balance is too low to access the Anthropic API…" (seen 2026-10-05) — open item O1.
_BILLING_TYPES = {"billing_error", "insufficient_quota"}
_BILLING_MESSAGE = re.compile(
    r"credit balance is too low|prepayment credits|insufficient (?:credit|funds|balance)|\bbilling\b",
    re.I,
)


def _error_type_and_message(exc: BaseException) -> tuple[Optional[str], str]:
    """Provider error type + message from the SDK error body, falling back to str(exc)."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else body
        return err.get("type") or err.get("code"), str(err.get("message") or exc)
    return None, str(exc)


def is_billing_error(exc: BaseException) -> bool:
    err_type, message = _error_type_and_message(exc)
    return (err_type in _BILLING_TYPES) or bool(_BILLING_MESSAGE.search(message))


def classify(exc: BaseException) -> ErrorKind:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "retryable"
    name = type(exc).__name__
    if name in ("APITimeoutError", "APIConnectionError", "ReadTimeout", "ConnectTimeout", "ConnectError"):
        return "retryable"
    status = error_status(exc)
    # Billing problems trip the breaker whatever status code carries them (O1).
    if status is not None and 400 <= status < 500 and is_billing_error(exc):
        return "auth_billing"
    if status is None:
        return "other"
    if status == 429 or status >= 500:
        return "retryable"
    if status in (401, 402, 403):
        return "auth_billing"
    if status == 404:
        return "not_found"
    if status == 400:
        return "bad_request"
    return "other"


def is_retryable(exc: BaseException) -> bool:
    return classify(exc) == "retryable"


def _keys(provider: str, model: Optional[str]) -> list[str]:
    return [provider] + ([f"{provider}:{model}"] if model else [])


def is_open(provider: str, model: Optional[str] = None) -> bool:
    """True while the provider (or this model of it) is being skipped."""
    now = time.monotonic()
    for key in _keys(provider, model):
        until = _open_until.get(key)
        if until is not None:
            if until > now:
                logger.debug("provider_health.skip key=%s remaining=%.0fs", key, until - now)
                return True
            del _open_until[key]
    return False


def record_failure(provider: str, model: Optional[str], exc: BaseException) -> ErrorKind:
    """Classify a provider failure and trip the breaker for auth/billing or missing-model errors."""
    kind = classify(exc)
    if kind in ("auth_billing", "not_found"):
        key = provider if kind == "auth_billing" else f"{provider}:{model}"
        already_open = _open_until.get(key, 0) > time.monotonic()
        _open_until[key] = time.monotonic() + BREAKER_SECONDS
        if not already_open:
            logger.error(
                "provider_health.breaker_open key=%s kind=%s status=%s — skipping for %.0f min: %s",
                key, kind, error_status(exc), BREAKER_SECONDS / 60, str(exc)[:300],
            )
    return kind


def reset() -> None:
    """Clear all breakers (tests)."""
    _open_until.clear()
