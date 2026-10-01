"""The real chat()/stream_chat() paths hand every provider the same prompt (Phase 3).

Uses tests/_prompt_harness.py (provider calls monkeypatched, no network).
Covers audit P5 (Gemini system_instruction + role turns, universal language
reminder), G14 (turn normalisation) and amendment 9 (the reminder is never
persisted, never sent to the guard, never seen by the summary-intent regex).
"""
import copy
from types import SimpleNamespace

import pytest

from _prompt_harness import MESSAGES, capture
from app.config import settings as app_settings
from app.prompts.builder import EARLIER_CONVERSATION, build_system_prompt, split_sections
from app.prompts.language import language_reminder
from app.routers.ai_assistant import _is_summary_intent, _inject_summary_rag
from app.schemas.ai_payloads import normalize_context, normalize_settings
from app.services.ai_service import AIService

pytestmark = pytest.mark.asyncio

CTX = {"student_mode": True, "grade": 8, "board": "ICSE", "subject": "Science", "language": "ta"}
SETTINGS = {"student_mode": True, "content_length": "detailed", "difficulty": "hard",
            "personality": "coach", "examples": True, "explain_3ways": False}
TA_REMINDER = language_reminder("ta")


async def test_every_provider_path_gets_the_same_system_prompt(monkeypatch):
    got = await capture(monkeypatch, CTX, SETTINGS)
    expected = build_system_prompt(normalize_context(CTX), normalize_settings(SETTINGS),
                                   question=MESSAGES[-1]["content"])
    for key in ("stream_gemini_system", "stream_claude_system", "chat_gemini_system", "chat_claude_system"):
        assert got[key] == expected, key


async def test_every_provider_path_gets_identical_role_turns_with_reminder(monkeypatch):
    got = await capture(monkeypatch, CTX, SETTINGS)
    expected = [
        {"role": "user", "content": MESSAGES[0]["content"]},
        {"role": "assistant", "content": MESSAGES[1]["content"]},
        {"role": "user", "content": f"{MESSAGES[2]['content']}\n\n{TA_REMINDER}"},
    ]
    assert got["stream_gemini_turns"] == expected
    assert got["stream_claude_messages"] == expected
    assert got["chat_claude_messages"] == expected
    # Gemini's non-stream path converts to google-genai Contents: assistant → "model".
    assert got["chat_gemini_turns"] == [{**t, "role": "model" if t["role"] == "assistant" else "user"} for t in expected]


async def test_openai_stream_gets_system_turns_reminder_and_params(monkeypatch):
    seen = {}

    class _Stream:
        def __aiter__(self):
            async def gen():
                yield SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason="stop", delta=SimpleNamespace(content="ok"))])
            return gen()

    async def fake_create(**kwargs):
        seen.update(kwargs)
        return _Stream()

    monkeypatch.setattr(app_settings, "CONTENT_GUARD_ENABLED", False)
    monkeypatch.setattr(AIService, "_get_gemini_async", lambda self: None)
    monkeypatch.setattr(AIService, "_get_openai",
                        lambda self: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))))
    out = "".join([t async for t in AIService().stream_chat(copy.deepcopy(MESSAGES), CTX, SETTINGS)])
    assert out == "ok"
    assert seen["messages"][0] == {"role": "system", "content": build_system_prompt(
        normalize_context(CTX), normalize_settings(SETTINGS), question=MESSAGES[-1]["content"])}
    assert seen["messages"][-1]["content"].endswith(TA_REMINDER)
    assert seen["max_tokens"] >= 512 and seen["temperature"] == 0.3   # science subject → factual


async def test_reminder_never_persisted_or_sent_to_guard(monkeypatch):
    """Amendment 9: the caller's history is untouched and the guard sees the raw turn."""
    guard_inputs = []

    async def fake_pipeline(self, query):
        guard_inputs.append(query)
        from app.services.content_guard import GuardAction, GuardResult
        return GuardResult(action=GuardAction.ALLOW, category="educational", message=None)

    monkeypatch.setattr("app.services.content_guard.ContentGuardService.run_input_pipeline", fake_pipeline)
    history = copy.deepcopy(MESSAGES)
    snapshot = copy.deepcopy(history)
    record = {}
    from _prompt_harness import _install_fakes
    _install_fakes(monkeypatch, record)
    monkeypatch.setattr(app_settings, "CONTENT_GUARD_ENABLED", True)
    async for _ in AIService().stream_chat(messages=history, context=CTX, chat_settings=SETTINGS):
        pass
    await AIService().chat(messages=history, context=CTX, chat_settings=SETTINGS)

    assert history == snapshot, "chat history (what the router saves) must not change"
    assert guard_inputs == [MESSAGES[-1]["content"]] * 2
    assert all(TA_REMINDER not in q for q in guard_inputs)
    assert record["stream_gemini_turns"][-1]["content"].endswith(TA_REMINDER)


async def test_reminder_text_does_not_trigger_summary_intent():
    """The router runs the summary regex on payload.message; the reminder can't affect it."""
    for lang in ("en", "hi", "ta", "ar"):
        assert not _is_summary_intent(language_reminder(lang))


async def test_summary_rag_path_gets_language_reminder(monkeypatch):
    """Phase 3: the summary wrapper had no language note; the universal reminder covers it."""
    messages = _inject_summary_rag([{"role": "user", "content": "summarize my notes"}],
                                   "summarize my notes", "Full document text …", file_name="notes.pdf")
    record = {}
    from _prompt_harness import _install_fakes
    _install_fakes(monkeypatch, record)
    async for _ in AIService().stream_chat(messages=messages, context={**CTX, "language": "hi"},
                                           chat_settings=SETTINGS, has_files=True):
        pass
    last = record["stream_claude_messages"][-1]["content"]
    assert "--- FULL DOCUMENT ---" in last and last.endswith(language_reminder("hi"))


async def test_history_starting_with_assistant_gets_neutral_placeholder(monkeypatch):
    import _prompt_harness as h
    monkeypatch.setattr(h, "MESSAGES", [
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Part one"},
        {"role": "user", "content": "Part two"},          # consecutive user turns (G14)
    ])
    got = await capture(monkeypatch, CTX, SETTINGS)
    turns = got["stream_claude_messages"]
    assert turns[0] == {"role": "user", "content": EARLIER_CONVERSATION}
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    assert turns[-1]["content"] == f"Part one\n\nPart two\n\n{TA_REMINDER}"


async def test_three_ways_history_reaches_the_prompt(monkeypatch):
    import _prompt_harness as h
    monkeypatch.setattr(h, "MESSAGES", [
        {"role": "user", "content": "What is inertia?"},
        {"role": "assistant", "content": "Inertia is…\n---\n**Analogy:** a bus braking…"},
        {"role": "user", "content": "And momentum?"},
    ])
    got = await capture(monkeypatch, CTX, SETTINGS)
    assert "Explain in 3 Ways is off for this answer" in split_sections(got["stream_claude_system"])["response_settings"]


async def test_student_mode_off_reaches_providers_without_profile(monkeypatch):
    got = await capture(monkeypatch, {**CTX, "student_mode": False}, {**SETTINGS, "student_mode": False})
    assert "<learner_profile>" not in got["stream_claude_system"]
    assert "Grade 8" not in got["stream_gemini_system"]


async def test_internal_generator_calls_get_no_reminder(monkeypatch):
    """chat() without context/settings (JSON generators) gets no language reminder."""
    record = {}
    from _prompt_harness import _install_fakes
    _install_fakes(monkeypatch, record)
    await AIService().chat([{"role": "user", "content": "Return JSON: [1,2]"}])
    assert record["chat_gemini_turns"] == [{"role": "user", "content": "Return JSON: [1,2]"}]
    assert set(split_sections(record["chat_gemini_system"])) == {"role", "hard_rules"}
