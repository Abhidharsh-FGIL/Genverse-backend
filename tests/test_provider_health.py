"""Provider retry policy, circuit breaker, G6 and content-guard fallback (no network).

Pre-Phase-3 items 3 and 4, and audit G6.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from app.config import settings as app_settings
from app.services import ai_service as ai_mod
from app.services import content_guard, provider_health
from app.services.ai_service import AIService

pytestmark = pytest.mark.asyncio

CTX = {"student_mode": True, "grade": 8, "board": "CBSE", "language": "en"}
SETTINGS = {"student_mode": True, "content_length": "summary"}
MSGS = [{"role": "user", "content": "What is photosynthesis?"}]


class StatusError(Exception):
    """Stands in for an SDK error carrying an HTTP status (openai/anthropic: status_code)."""

    def __init__(self, status: int, msg: str = "boom"):
        super().__init__(f"Error code: {status} - {msg}")
        self.status_code = status


class GenaiError(Exception):
    """google-genai style: `.code` attribute, message starts with the status."""

    def __init__(self, code: int):
        super().__init__(f"{code} RESOURCE_EXHAUSTED. credits depleted")
        self.code = code


@pytest.fixture(autouse=True)
def _clean_breaker(monkeypatch):
    provider_health.reset()
    monkeypatch.setattr(app_settings, "CONTENT_GUARD_ENABLED", False)
    monkeypatch.setattr(ai_mod.asyncio, "sleep", _no_sleep)
    yield
    provider_health.reset()


async def _no_sleep(*_a, **_k):
    return None


# ── classification ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("exc, kind", [
    (StatusError(429), "retryable"), (StatusError(500), "retryable"), (StatusError(503), "retryable"),
    (StatusError(400), "bad_request"), (StatusError(401), "auth_billing"), (GenaiError(402), "auth_billing"),
    (StatusError(403), "auth_billing"), (StatusError(404), "not_found"), (asyncio.TimeoutError(), "retryable"),
    (Exception("402 RESOURCE_EXHAUSTED. Your prepayment credits are depleted."), "auth_billing"),
    (ValueError("no status here"), "other"),
])
async def test_classify(exc, kind):
    assert provider_health.classify(exc) == kind
    assert provider_health.is_retryable(exc) == (kind == "retryable")


async def test_old_402_is_no_longer_retryable():
    """The old substring check retried 402 RESOURCE_EXHAUSTED 4× before falling back."""
    assert not ai_mod._is_retryable(GenaiError(402))


# ── breaker ──────────────────────────────────────────────────────────────────

async def test_breaker_opens_once_with_one_error_log(caplog):
    with caplog.at_level(logging.ERROR, logger="app.services.provider_health"):
        provider_health.record_failure("gemini", "m1", GenaiError(402))
        provider_health.record_failure("gemini", "m1", GenaiError(402))
    assert provider_health.is_open("gemini") and provider_health.is_open("gemini", "other-model")
    assert len([r for r in caplog.records if "breaker_open" in r.getMessage()]) == 1


async def test_not_found_opens_only_that_model():
    provider_health.record_failure("claude", "old-model", StatusError(404))
    assert provider_health.is_open("claude", "old-model")
    assert not provider_health.is_open("claude", "new-model")


async def test_breaker_expires(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(provider_health.time, "monotonic", lambda: now[0])
    provider_health.record_failure("openai", "m", StatusError(401))
    assert provider_health.is_open("openai")
    now[0] += provider_health.BREAKER_SECONDS + 1
    assert not provider_health.is_open("openai")


async def test_retryable_and_bad_request_do_not_open_breaker():
    provider_health.record_failure("openai", "m", StatusError(503))
    provider_health.record_failure("openai", "m", StatusError(400))
    assert not provider_health.is_open("openai", "m")


# ── Gemini retry helper (non-stream) ─────────────────────────────────────────

def _genai_client(fail_with: dict[str, list]):
    """fail_with: model → list of exceptions to raise in order, then succeed."""
    calls = []

    async def generate_content(model, contents, config=None):
        calls.append(model)
        queue = fail_with.get(model, [])
        if queue:
            raise queue.pop(0)
        return SimpleNamespace(text=f"ok from {model}")

    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))), calls


async def test_gemini_402_goes_straight_to_next_provider_and_trips_breaker(monkeypatch):
    monkeypatch.setattr(app_settings, "AI_GEMINI_BACKUP_MODEL", "backup")
    client, calls = _genai_client({"primary": [GenaiError(402)]})
    with pytest.raises(GenaiError):
        await ai_mod._gemini_generate_with_retry(client, "primary", "x")
    assert calls == ["primary"], "402: no retry and no backup model (same key)"
    with pytest.raises(Exception):
        await ai_mod._gemini_generate_with_retry(client, "primary", "x")
    assert calls == ["primary"], "breaker open: Gemini not called again"


async def test_gemini_503_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(app_settings, "AI_GEMINI_BACKUP_MODEL", "backup")
    client, calls = _genai_client({"primary": [StatusError(503)]})
    r = await ai_mod._gemini_generate_with_retry(client, "primary", "x")
    assert r.text == "ok from primary" and calls == ["primary", "primary"]


async def test_gemini_404_on_primary_uses_backup(monkeypatch):
    monkeypatch.setattr(app_settings, "AI_GEMINI_BACKUP_MODEL", "backup")
    client, calls = _genai_client({"primary": [StatusError(404)]})
    r = await ai_mod._gemini_generate_with_retry(client, "primary", "x")
    assert r.text == "ok from backup" and calls == ["primary", "backup"]


# ── streaming fallback chain: G6 and breaker ─────────────────────────────────

def _install_streams(monkeypatch, gemini=None, openai=None, claude=None):
    """Each arg: async generator function (self, system, turns, params) or None (not configured)."""
    for name, getter, fn in (("gemini", "_get_gemini_async", gemini), ("openai", "_get_openai", openai),
                             ("claude", "_get_anthropic", claude)):
        monkeypatch.setattr(AIService, getter, (lambda self: object()) if fn else (lambda self: None))
        if fn:
            monkeypatch.setattr(AIService, f"_stream_{'anthropic' if name == 'claude' else name}", fn)


async def _collect(**kw):
    return "".join([t async for t in AIService().stream_chat(MSGS, CTX, SETTINGS, **kw)])


async def test_g6_no_fallback_after_partial_output(monkeypatch):
    openai_called = []

    async def gemini(self, *a):
        yield "Photosynthesis is "
        raise StatusError(503)

    async def openai(self, *a):
        openai_called.append(True)
        yield "A COMPLETE SECOND ANSWER"

    _install_streams(monkeypatch, gemini=gemini, openai=openai)
    out = await _collect()
    assert out.startswith("Photosynthesis is ")
    assert out.endswith(AIService.error_message("generation_interrupted", "en"))
    assert not openai_called, "G6: must not stream a second answer after partial output"


async def test_failure_before_any_output_falls_back(monkeypatch):
    async def gemini(self, *a):
        raise GenaiError(402)
        yield  # pragma: no cover

    async def openai(self, *a):
        yield "fallback answer"

    _install_streams(monkeypatch, gemini=gemini, openai=openai)
    assert await _collect() == "fallback answer"


async def test_document_path_skips_claude_when_breaker_open(monkeypatch):
    claude_calls = []

    async def claude(self, *a):
        claude_calls.append(True)
        raise StatusError(404, "model not found")
        yield  # pragma: no cover

    async def openai(self, *a):
        yield "openai answer"

    _install_streams(monkeypatch, openai=openai, claude=claude)
    assert await _collect(has_files=True) == "openai answer"
    assert await _collect(has_files=True) == "openai answer"
    assert claude_calls == [True], "after a 404 the model is skipped for BREAKER_SECONDS"


async def test_all_providers_unavailable_gives_localized_message(monkeypatch):
    _install_streams(monkeypatch)
    out = "".join([t async for t in AIService().stream_chat(MSGS, {**CTX, "language": "ta"}, SETTINGS)])
    assert out == AIService.error_message("temporarily_unavailable", "ta")


# ── content guard: fallback classifier and fail-open warnings (item 4) ───────

async def test_guard_uses_fallback_provider_when_primary_fails(monkeypatch):
    async def gemini_fail(**kw):
        raise GenaiError(402)

    async def openai_ok(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"category": "educational", "confidence": 0.9, "reason": "school topic"}'))])

    monkeypatch.setattr(content_guard, "_get_guard_client",
                        lambda: SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=gemini_fail))))
    monkeypatch.setattr(content_guard, "_get_guard_openai_client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=openai_ok))))
    monkeypatch.setattr(app_settings, "CONTENT_GUARD_LLM_CLASSIFIER", True)
    result = await content_guard.ContentGuardService().classify_query("what is photosynthesis")
    assert result.category == "educational" and result.confidence == 0.9


async def test_guard_fail_open_logs_warning_with_reason(monkeypatch, caplog):
    async def fail(**kw):
        raise StatusError(500)

    monkeypatch.setattr(content_guard, "_get_guard_client",
                        lambda: SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=fail))))
    monkeypatch.setattr(content_guard, "_get_guard_openai_client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail))))
    monkeypatch.setattr(app_settings, "CONTENT_GUARD_LLM_CLASSIFIER", True)
    with caplog.at_level(logging.WARNING, logger="content_guard"):
        result = await content_guard.ContentGuardService().classify_query("what is photosynthesis")
    assert result.action == content_guard.GuardAction.ALLOW
    msgs = [r.getMessage() for r in caplog.records if "content_guard.fail_open" in r.getMessage()]
    assert len(msgs) == 1 and "gemini" in msgs[0] and "openai" in msgs[0]


async def test_stream_guard_timeout_logs_fail_open(monkeypatch, caplog):
    async def slow_pipeline(self, query):
        await asyncio.Event().wait()   # never answers

    async def openai(self, *a):
        yield "answer"

    monkeypatch.setattr(app_settings, "CONTENT_GUARD_ENABLED", True)
    monkeypatch.setattr(content_guard.ContentGuardService, "run_input_pipeline", slow_pipeline)
    monkeypatch.setattr(ai_mod, "GUARD_BUDGET_S", 0.01)
    _install_streams(monkeypatch, openai=openai)
    with caplog.at_level(logging.WARNING, logger="content_guard"):
        assert await _collect() == "answer"
    assert any("content_guard.fail_open reason=timeout" in r.getMessage() for r in caplog.records)


# ── localized messages (Phase 3) ─────────────────────────────────────────────

@pytest.mark.parametrize("lang", ["en", "ar", "hi", "ta"])
async def test_messages_exist_in_every_ui_language(lang):
    for key in AIService._ERROR_MESSAGES:
        assert lang in AIService._ERROR_MESSAGES[key], key
    for table in (content_guard.BLOCK_MESSAGES["harmful"], content_guard.BLOCK_MESSAGES["sensitive"],
                  content_guard.REDIRECT_MESSAGES["non_educational"], content_guard.OUTPUT_REPLACEMENT_MESSAGE):
        assert lang in table


# ── decision 1 / decision 2 (pre-Phase-4) ────────────────────────────────────

async def test_guard_budget_widens_only_while_gemini_breaker_open():
    assert ai_mod._guard_budget() == ai_mod.GUARD_BUDGET_S == 0.5
    provider_health.record_failure("gemini", app_settings.AI_PRIMARY_MODEL, GenaiError(402))
    assert ai_mod._guard_budget() == ai_mod.GUARD_BUDGET_FALLBACK_S == 1.5


async def test_gemini_temperature_per_model_family(monkeypatch):
    """No temperature for Gemini 3.x (primary); the 2.x backup model still gets it."""
    from google.genai import types as gt
    seen = {}

    async def generate_content(model, contents, config=None):
        seen[model] = getattr(config, "temperature", None)
        if model == "gemini-3.5-flash":
            raise StatusError(503)
        return SimpleNamespace(text="ok")

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    monkeypatch.setattr(app_settings, "AI_GEMINI_BACKUP_MODEL", "gemini-2.5-flash-lite")
    base = gt.GenerateContentConfig(system_instruction="sys")
    await ai_mod._gemini_generate_with_retry(client, "gemini-3.5-flash", "x", base, temperature=0.7)
    assert seen == {"gemini-3.5-flash": None, "gemini-2.5-flash-lite": 0.7}
    assert base.temperature is None, "shared config must not be mutated"


# ── O1: billing errors carried by HTTP 400 (Anthropic "credit balance is too low") ──

def _anthropic_400(message: str, err_type: str = "invalid_request_error"):
    import anthropic
    import httpx
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": err_type, "message": message}}
    return anthropic.BadRequestError(f"Error code: 400 - {body}", response=httpx.Response(400, request=req), body=body)


async def test_o1_anthropic_credit_balance_400_is_billing():
    exc = _anthropic_400("Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing.")
    assert provider_health.classify(exc) == "auth_billing"
    provider_health.record_failure("claude", "claude-sonnet-5-5", exc)
    assert provider_health.is_open("claude") and provider_health.is_open("claude", "any-other-model")


async def test_o1_billing_error_type_is_billing_even_without_message():
    assert provider_health.classify(_anthropic_400("Request blocked.", err_type="billing_error")) == "auth_billing"


async def test_o1_ordinary_400_is_still_bad_request():
    exc = _anthropic_400("max_tokens: Input should be a valid integer")
    assert provider_health.classify(exc) == "bad_request"
    provider_health.record_failure("claude", "m", exc)
    assert not provider_health.is_open("claude")


async def test_o1_document_path_skips_claude_after_billing_400(monkeypatch):
    claude_calls = []

    async def claude(self, *a):
        claude_calls.append(True)
        raise _anthropic_400("Your credit balance is too low to access the Anthropic API.")
        yield  # pragma: no cover

    async def openai(self, *a):
        yield "openai answer"

    _install_streams(monkeypatch, openai=openai, claude=claude)
    assert await _collect(has_files=True) == "openai answer"
    assert await _collect(has_files=True) == "openai answer"
    assert claude_calls == [True], "billing 400 must trip the breaker: Claude not called again"
