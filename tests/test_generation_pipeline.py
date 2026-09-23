r"""The generation pipeline's LaTeX handling end to end, with the LLM mocked.

No database, no network: the model call is replaced with canned responses, so
these run anywhere. They cover the two halves of the fix that the pure-regex
tests cannot — finalize_generated_questions flagging behaviour, and the retry
that re-asks the model when a question is still broken.
"""
import json

import pytest

from app.services.ai_service import AIService

pytestmark = pytest.mark.asyncio


OVER_ESCAPED_RAW = [{
    "id": "q10", "type": "fill", "subtype": None,
    "text": r"Incline $\\theta = 37^\\circ$, mass $m = 5\\,\\mathrm{kg}$, $\\mu_s$ ___.",
    "options": None, "pairs": None,
    "correct_answer": r"$\\tan\\theta$",
    "explanation": r"Using $\\frac{f}{N}$ and $50\\,\\mathrm{N}$.",
    "marks": 1, "blooms_level": "apply",
}]


async def test_finalize_repairs_over_escaping_and_does_not_flag():
    qj, ak = await AIService.finalize_generated_questions(OVER_ESCAPED_RAW, {"fill"})
    text = qj[0]["text"]
    assert r"\\theta" not in text and r"\theta" in text
    assert r"\\," not in text and r"\," in text
    assert r"\\circ" not in text and r"\circ" in text
    assert r"\\tan" not in ak[0]["correctAnswer"]
    assert r"\\frac" not in ak[0]["explanation"]
    assert not qj[0].get("needs_review")


async def test_finalize_preserves_matrix_line_breaks():
    raw = [{"id": "m1", "type": "fill",
            "text": r"$A = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix}$ gives ___.",
            "correct_answer": "5", "explanation": "", "marks": 1}]
    qj, _ = await AIService.finalize_generated_questions(raw, {"fill"})
    assert qj[0]["text"] == raw[0]["text"]
    assert not qj[0].get("needs_review")


async def test_finalize_flags_but_keeps_genuinely_broken_math():
    raw = [{"id": "b1", "type": "fill", "text": r"Evaluate $\frac{1}{2$ now ___.",
            "correct_answer": "x", "explanation": "", "marks": 1}]
    qj, _ = await AIService.finalize_generated_questions(raw, {"fill"})
    assert len(qj) == 1, "a flagged question must be kept, not dropped"
    assert qj[0]["needs_review"] is True
    assert qj[0]["review_reason"]


async def test_retry_replaces_a_flagged_question_when_the_model_fixes_it(monkeypatch):
    broken = [{"id": "b1", "type": "fill", "text": r"Evaluate $\frac{1}{2$ now ___.",
               "correct_answer": "x", "explanation": "", "marks": 1}]
    qj, ak = await AIService.finalize_generated_questions(broken, {"fill"})
    assert qj[0]["needs_review"] is True

    calls = []

    async def fake_chat(self, messages, **kwargs):
        calls.append(messages[0]["content"])
        return json.dumps([{
            "id": "b1", "type": "fill", "text": r"Evaluate $\frac{1}{2}$ now ___.",
            "options": None, "pairs": None, "correct_answer": "x",
            "explanation": "", "marks": 1, "blooms_level": None,
        }])

    monkeypatch.setattr(AIService, "chat", fake_chat)
    ai = AIService()
    qj, ak = await ai.repair_flagged_questions(qj, ak, {"fill"})

    assert len(calls) == 1, "one retry should have sufficed"
    assert not qj[0].get("needs_review")
    assert qj[0]["text"] == r"Evaluate $\frac{1}{2}$ now ___."
    # The retry prompt must pin the content, not invite a rewrite.
    assert "_problem" in calls[0]
    assert "Rewrite ONLY the mathematical notation" in calls[0]


async def test_retry_gives_up_after_max_attempts_and_keeps_the_flag(monkeypatch):
    broken = [{"id": "b1", "type": "fill", "text": r"Evaluate $\frac{1}{2$ now ___.",
               "correct_answer": "x", "explanation": "", "marks": 1}]
    qj, ak = await AIService.finalize_generated_questions(broken, {"fill"})

    calls = []

    async def never_fixes(self, messages, **kwargs):
        calls.append(1)
        return json.dumps([{
            "id": "b1", "type": "fill", "text": r"Still $\frac{1}{2$ broken ___.",
            "options": None, "pairs": None, "correct_answer": "x",
            "explanation": "", "marks": 1, "blooms_level": None,
        }])

    monkeypatch.setattr(AIService, "chat", never_fixes)
    ai = AIService()
    qj, ak = await ai.repair_flagged_questions(qj, ak, {"fill"}, max_attempts=2)

    assert len(calls) == 2, "must stop at max_attempts"
    assert len(qj) == 1, "must still be saved, not dropped"
    assert qj[0]["needs_review"] is True


async def test_retry_is_a_noop_when_nothing_is_flagged(monkeypatch):
    async def must_not_be_called(self, messages, **kwargs):
        raise AssertionError("no model call should happen when nothing is flagged")
    monkeypatch.setattr(AIService, "chat", must_not_be_called)
    qj, ak = await AIService.finalize_generated_questions(OVER_ESCAPED_RAW, {"fill"})
    ai = AIService()
    out_q, out_a = await ai.repair_flagged_questions(qj, ak, {"fill"})
    assert out_q == qj and out_a == ak


async def test_retry_survives_a_model_failure(monkeypatch):
    broken = [{"id": "b1", "type": "fill", "text": r"Evaluate $\frac{1}{2$ now ___.",
               "correct_answer": "x", "explanation": "", "marks": 1}]
    qj, ak = await AIService.finalize_generated_questions(broken, {"fill"})

    async def explode(self, messages, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(AIService, "chat", explode)
    ai = AIService()
    qj, ak = await ai.repair_flagged_questions(qj, ak, {"fill"})
    assert len(qj) == 1 and qj[0]["needs_review"] is True
