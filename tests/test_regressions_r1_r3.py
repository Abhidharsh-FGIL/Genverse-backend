"""Regressions R1–R3 from REGRESSION_CHECK.md: non-assistant features keep their old behaviour.

R1  Playground (and any caller) system-role turns reach the model again — as a
    <task> section in the system prompt, for every provider path.
R2  Only the AI Assistant (chat settings present) gets settings-driven
    max_tokens / temperature; every other caller sends neither, as before.
R3  The emoji ban is an AI Assistant rule only.
No network: providers are faked.
"""
from types import SimpleNamespace

import pytest

from _prompt_harness import capture, _install_fakes
from app.config import settings as app_settings
from app.prompts.builder import build_system_prompt, caller_task, split_sections, to_provider_messages
from app.prompts.params import GenParams, generation_params
from app.schemas.ai_payloads import normalize_context, normalize_settings
from app.services.ai_service import AIService

pytestmark = pytest.mark.asyncio

SESSION_CTX = {"student_mode": True, "grade": 8, "board": "CBSE", "language": "en"}
ASSISTANT_SETTINGS = {"student_mode": True, "content_length": "summary"}
MODE_PROMPT_MARK = "thought experiment about 'Reflection of light'"


def _recording_openai(monkeypatch):
    """Fake OpenAI client (stream + non-stream) that records every create() kwargs."""
    calls = []

    class _Stream:
        def __aiter__(self):
            async def gen():
                yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
                    finish_reason="stop", delta=SimpleNamespace(content="ok"))])
            return gen()

    async def create(**kw):
        calls.append(kw)
        if kw.get("stream"):
            return _Stream()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))])

    monkeypatch.setattr(app_settings, "CONTENT_GUARD_ENABLED", False)
    monkeypatch.setattr(AIService, "_get_gemini_async", lambda self: None)
    monkeypatch.setattr(AIService, "_get_anthropic", lambda self: None)
    monkeypatch.setattr(AIService, "_get_openai",
                        lambda self: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    return calls


async def _drain(gen):
    return "".join([c async for c in gen])


# ── R1 ───────────────────────────────────────────────────────────────────────

async def test_r1_playground_mode_prompt_reaches_gemini_and_claude(monkeypatch):
    record = {}
    _install_fakes(monkeypatch, record)
    await _drain(AIService().stream_playground(
        topic="Reflection of light", mode="experiment", messages=[], grade=8, harder_mode=False,
        context=dict(SESSION_CTX)))
    task = split_sections(record["stream_gemini_system"])["task"]
    assert MODE_PROMPT_MARK in task and "Adapt for Grade 8 students." in task
    assert all(t["role"] in ("user", "assistant") for t in record["stream_gemini_turns"])


async def test_r1_playground_mode_prompt_reaches_openai(monkeypatch):
    calls = _recording_openai(monkeypatch)
    await _drain(AIService().stream_playground(
        topic="Reflection of light", mode="challenge", messages=[], grade=8, harder_mode=True,
        context=dict(SESSION_CTX)))
    system = calls[0]["messages"][0]
    assert system["role"] == "system" and "challenging problem about 'Reflection of light'" in system["content"]
    assert "Make it harder than usual." in system["content"]


async def test_r1_system_turns_in_non_user_chat_calls_are_kept(monkeypatch):
    record = {}
    _install_fakes(monkeypatch, record)
    await AIService().chat([{"role": "system", "content": "You are a JSON API."},
                            {"role": "user", "content": "Return []"}])
    s = split_sections(record["chat_gemini_system"])
    assert "You are a JSON API." in s["task"]
    assert record["chat_gemini_turns"] == [{"role": "user", "content": "Return []"}]


async def test_r1_assistant_prompt_has_no_task_section(monkeypatch):
    got = await capture(monkeypatch, SESSION_CTX, ASSISTANT_SETTINGS)
    for key in ("stream_gemini_system", "stream_claude_system", "chat_gemini_system", "chat_claude_system"):
        assert "<task>" not in got[key], key


async def test_r1_caller_task_helper():
    assert caller_task([{"role": "user", "content": "x"}]) is None
    assert caller_task([{"role": "system", "content": " a "}, {"role": "system", "content": "b"}]) == "a\n\nb"
    assert to_provider_messages([{"role": "system", "content": "a"}, {"role": "user", "content": "q"}], None) == \
        [{"role": "user", "content": "q"}]


# ── R2 ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ctx", [{"stream": "Science", "language": "en"},          # career analysis
                                 {"grade": 8, "board": "CBSE", "language": "hi"},  # audio Q&A
                                 dict(SESSION_CTX)])                               # vault query / playground
async def test_r2_context_callers_without_settings_get_provider_defaults(ctx):
    assert generation_params(normalize_context(ctx), None, "q", False, True) == GenParams()


async def test_r2_career_analysis_sends_no_ceiling_or_temperature(monkeypatch):
    calls = _recording_openai(monkeypatch)
    await AIService().analyze_career(interests=["robotics"], strengths=["maths"], target_careers=None,
                                     grade=11, context={"stream": "Science"}, language="hi")
    assert "max_tokens" not in calls[0] and "temperature" not in calls[0]


async def test_r2_playground_stream_sends_no_ceiling_or_temperature(monkeypatch):
    calls = _recording_openai(monkeypatch)
    await _drain(AIService().stream_playground(topic="Light", mode="play", messages=[], grade=8,
                                               harder_mode=False, context=dict(SESSION_CTX)))
    assert "max_tokens" not in calls[0] and "temperature" not in calls[0]


async def test_r2_vault_query_and_audio_send_no_ceiling(monkeypatch):
    calls = _recording_openai(monkeypatch)
    await AIService().ask_document("What does it say?", "Doc text.", dict(SESSION_CTX))
    await AIService().chat([{"role": "user", "content": "What is light?"}], context={"language": "hi"})
    assert all("max_tokens" not in c and "temperature" not in c for c in calls)


async def test_r2_assistant_still_gets_settings_driven_params(monkeypatch):
    calls = _recording_openai(monkeypatch)
    await _drain(AIService().stream_chat([{"role": "user", "content": "What is light?"}],
                                         dict(SESSION_CTX), {**ASSISTANT_SETTINGS, "personality": "friend"}))
    assert calls[0]["max_tokens"] >= 512 and calls[0]["temperature"] == 0.7


# ── R3 ───────────────────────────────────────────────────────────────────────

async def test_r3_emoji_ban_only_in_assistant_prompt():
    ctx, cs = normalize_context(SESSION_CTX), normalize_settings(ASSISTANT_SETTINGS)
    assert "emoji" not in build_system_prompt(None, None).lower()            # internal generators
    assert "emoji" not in build_system_prompt(ctx, None).lower()             # career / audio / vault / playground
    assistant = build_system_prompt(ctx, cs)
    assert assistant.lower().count("emoji") == 1 and "7. Do not use emojis." in split_sections(assistant)["hard_rules"]


async def test_r3_playground_generators_get_no_emoji_ban(monkeypatch):
    record = {}
    _install_fakes(monkeypatch, record)
    monkeypatch.setattr(AIService, "_get_openai", lambda self: None)
    await AIService().playground_setup(topic="Photosynthesis")
    assert "emoji" not in record["chat_gemini_system"].lower()
