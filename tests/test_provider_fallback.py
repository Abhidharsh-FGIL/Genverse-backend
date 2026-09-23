r"""Provider redundancy on the JEE/NEET exam-track generation path.

That path used Gemini's structured output exclusively and `return []` the
moment it raised — no fallback — while the standard path went through
self.chat() and its Gemini -> OpenAI -> Anthropic chain. A Gemini billing 402
therefore took out every JEE/NEET generation, producing this in production:

    [ExamAssessment:jee] Pass 1 failed: 402 RESOURCE_EXHAUSTED
    [Assessment-Celery] LLM returned 0 raw questions
    Task generate_assessment_task[...] succeeded in 2.33s

...and the SSE stream then reported stage="complete" with the message
"0 questions generated successfully!" — a success message for a total failure.

No network: the provider calls are monkeypatched.
"""
import json

import pytest

from app.services.ai_service import AIService

pytestmark = pytest.mark.asyncio


FIVE_QUESTIONS = json.dumps([
    {"id": f"q{i}", "type": "mcq", "subtype": "standard",
     "text": f"Question {i} about $\\ce{{H2SO4}}$ and $\\theta$?",
     "options": ["a", "b", "c", "d"], "pairs": None,
     "correct_answer": "a", "explanation": "because.", "marks": 4,
     "blooms_level": "apply", "requires_question_image": False}
    for i in range(1, 6)
])


class _Boom(Exception):
    """Stands in for google.genai's 402 RESOURCE_EXHAUSTED."""


@pytest.fixture
def no_gemini(monkeypatch):
    """Gemini raises, exactly as it does when prepayment credits are depleted."""
    async def explode(*a, **k):
        raise _Boom("402 RESOURCE_EXHAUSTED. Your prepayment credits are depleted.")
    monkeypatch.setattr("app.services.ai_service._gemini_generate_with_retry", explode,
                        raising=False)
    monkeypatch.setattr(AIService, "_get_gemini_async", lambda self: object())


async def test_exam_track_falls_back_when_gemini_fails(no_gemini, monkeypatch):
    calls = []

    async def fake_chat(self, messages, **kwargs):
        calls.append(messages[0]["content"])
        return FIVE_QUESTIONS

    monkeypatch.setattr(AIService, "chat", fake_chat)

    questions = await AIService().generate_practice_assessment(
        subject="Chemistry", topics=["Chemical Bonding"], grade=12, board="CBSE",
        difficulty="medium", question_count=5, question_types=["mcq"],
        mode="practice", exam_type="jee",
    )

    assert len(questions) == 5, "the fallback must rescue the generation"
    assert len(calls) == 1, "chat() should be called exactly once as the fallback"
    # the fallback must receive the real exam prompt, not a stub
    assert "JEE" in calls[0] or "jee" in calls[0].lower()


async def test_exam_track_returns_empty_only_when_every_provider_fails(no_gemini, monkeypatch):
    async def also_explode(self, messages, **kwargs):
        raise _Boom("all providers down")

    monkeypatch.setattr(AIService, "chat", also_explode)

    questions = await AIService().generate_practice_assessment(
        subject="Chemistry", topics=["Chemical Bonding"], grade=12, board="CBSE",
        difficulty="medium", question_count=5, question_types=["mcq"],
        mode="practice", exam_type="jee",
    )
    assert questions == [], "must fail cleanly, not raise"


async def test_exam_track_falls_back_when_gemini_returns_unparseable_text(monkeypatch):
    """Not just exceptions: a 200 response with no usable JSON must also fall back."""
    class _Resp:
        text = "I'm sorry, I can't help with that request."

    async def empty(*a, **k):
        return _Resp()

    monkeypatch.setattr("app.services.ai_service._gemini_generate_with_retry", empty,
                        raising=False)
    monkeypatch.setattr(AIService, "_get_gemini_async", lambda self: object())

    async def fake_chat(self, messages, **kwargs):
        return FIVE_QUESTIONS

    monkeypatch.setattr(AIService, "chat", fake_chat)

    questions = await AIService().generate_practice_assessment(
        subject="Chemistry", topics=["Bonding"], grade=12, board="CBSE",
        difficulty="medium", question_count=5, question_types=["mcq"],
        mode="practice", exam_type="jee",
    )
    assert len(questions) == 5


async def test_missing_gemini_client_still_falls_back(monkeypatch):
    """A missing key used to be indistinguishable from an empty response: no
    log line, no fallback, just []."""
    monkeypatch.setattr(AIService, "_get_gemini_async", lambda self: None)

    async def fake_chat(self, messages, **kwargs):
        return FIVE_QUESTIONS

    monkeypatch.setattr(AIService, "chat", fake_chat)

    questions = await AIService().generate_practice_assessment(
        subject="Chemistry", topics=["Bonding"], grade=12, board="CBSE",
        difficulty="medium", question_count=5, question_types=["mcq"],
        mode="practice", exam_type="jee",
    )
    assert len(questions) == 5


async def test_standard_path_still_uses_chat_directly(monkeypatch):
    """The non-exam path must be unaffected by the exam-track change."""
    calls = []

    async def fake_chat(self, messages, **kwargs):
        calls.append(1)
        return FIVE_QUESTIONS

    monkeypatch.setattr(AIService, "chat", fake_chat)
    questions = await AIService().generate_practice_assessment(
        subject="Physics", topics=["Kinematics"], grade=12, board="CBSE",
        difficulty="medium", question_count=5, question_types=["mcq"],
        mode="practice", exam_type=None,
    )
    assert len(questions) == 5
    assert len(calls) == 1
