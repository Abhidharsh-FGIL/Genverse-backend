"""O2: turning Explain-in-3-Ways off mid-chat stops the format on the next reply.

The previous answers' 3-ways blocks are removed from what the model sees (the
stored history is not changed), and the "off for this answer" line is still
emitted. No network: providers faked via tests/_prompt_harness.py.
"""
import copy

import pytest

import _prompt_harness as h
from _prompt_harness import capture
from app.prompts.builder import split_sections, strip_three_ways

pytestmark = pytest.mark.asyncio

BLOCK = ("\n\n---\n\n**Analogy:** Like a kitchen making food.\n\n**Technical:** 6CO2 + 6H2O → C6H12O6 + 6O2."
         "\n\n**Real-World Example:** A potted plant on a window sill.")
HISTORY = [
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Plants make food from light." + BLOCK},
    {"role": "user", "content": "What is respiration in plants?"},
    {"role": "assistant", "content": "Plants release energy from glucose." + BLOCK},
    {"role": "user", "content": "What is transpiration?"},
]
CTX = {"student_mode": True, "grade": 5, "board": "CBSE", "language": "en"}
OFF = {"student_mode": True, "content_length": "summary", "explain_3ways": False}
ON = {**OFF, "explain_3ways": True}


async def test_strip_removes_block_and_keeps_answer():
    out = strip_three_ways(HISTORY)
    assert out[1]["content"] == "Plants make food from light."
    assert out[3]["content"] == "Plants release energy from glucose."
    assert out[0] == HISTORY[0] and out[4] == HISTORY[4]


async def test_strip_does_not_mutate_and_ignores_unrelated_text():
    hist = copy.deepcopy(HISTORY)
    strip_three_ways(hist)
    assert hist == HISTORY
    lone = [{"role": "assistant", "content": "An **Analogy:** helps here, nothing else."}]
    assert strip_three_ways(lone) == lone          # no second 3-ways label → untouched
    user = [{"role": "user", "content": "Explain with **Analogy:** and **Technical:** parts"}]
    assert strip_three_ways(user) == user          # user turns are never changed


async def test_three_ways_off_mid_chat_hides_block_from_every_provider_path(monkeypatch):
    monkeypatch.setattr(h, "MESSAGES", copy.deepcopy(HISTORY))
    got = await capture(monkeypatch, CTX, OFF)
    for key in ("stream_gemini_turns", "stream_claude_messages", "chat_claude_messages", "chat_gemini_turns"):
        joined = "\n".join(t["content"] for t in got[key])
        assert "**Analogy:**" not in joined and "**Technical:**" not in joined, key
    assert "Explain in 3 Ways is off for this answer" in split_sections(got["stream_claude_system"])["response_settings"]
    assert h.MESSAGES == HISTORY, "stored history must be untouched"


async def test_three_ways_on_keeps_previous_blocks(monkeypatch):
    monkeypatch.setattr(h, "MESSAGES", copy.deepcopy(HISTORY))
    got = await capture(monkeypatch, CTX, ON)
    assert "**Analogy:**" in "\n".join(t["content"] for t in got["stream_claude_messages"])
