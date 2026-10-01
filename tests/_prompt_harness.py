"""Capture the exact prompts AIService sends to each provider — no network.

Used by tests/test_prompt_providers.py. (The Phase 1 golden baseline in
tests/golden/archive/ was captured with an earlier version of this harness,
when Gemini still received a single flattened prompt.) Provider entry points
are monkeypatched to record their inputs and return a fixed reply; the content
guard is disabled so no classifier call is made.
"""
import itertools
import random
from types import SimpleNamespace

from app.config import settings as app_settings
from app.services import ai_service as ai_mod
from app.services.ai_service import AIService

MESSAGES = [
    {"role": "user", "content": "What is photosynthesis?"},
    {"role": "assistant", "content": "Plants make food from light."},
    {"role": "user", "content": "Explain the equation, please."},
]


def build_combos() -> list[dict]:
    """Deterministic set of valid (non-legacy) context × settings combinations."""
    rng = random.Random(20261001)
    grades = [None, 2, 5, 8, 11]
    boards = [None, "CBSE", "ICSE", "State Board", "IB", "Cambridge", "Kerala Board"]
    languages = [None, "en", "hi", "ta", "ar"]
    subjects = [None, "Science"]
    personalities = ["mentor", "coach", "professor", "friend", "technical-expert"]
    difficulties = ["easy", "medium", "hard", "expert"]
    lengths = ["small", "summary", "detailed", "deep-dive"]
    output_modes = ["text", "structured", "bullets"]

    combos = []
    # Every grade × board pair once, other fields randomised.
    for grade, board in itertools.product(grades, boards):
        combos.append(_combo(rng, grade, board, languages, subjects, personalities,
                             difficulties, lengths, output_modes))
    # Every language, personality, difficulty and length at least once more.
    for lang in languages:
        c = _combo(rng, 8, "CBSE", languages, subjects, personalities, difficulties, lengths, output_modes)
        c["context"]["language"] = lang
        combos.append(c)
    for p, d, l in zip(personalities, difficulties * 2, lengths * 2):
        c = _combo(rng, 5, "ICSE", languages, subjects, personalities, difficulties, lengths, output_modes)
        c["settings"].update(personality=p, difficulty=d, content_length=l)
        combos.append(c)
    return combos


def _combo(rng, grade, board, languages, subjects, personalities, difficulties, lengths, output_modes):
    context = {
        "student_mode": rng.choice([True, False]),
        "grade": grade,
        "board": board,
        "subject": rng.choice(subjects),
        "language": rng.choice(languages),
    }
    chat_settings = {
        "difficulty": rng.choice(difficulties),
        "personality": rng.choice(personalities),
        "content_length": rng.choice(lengths),
        "explain_3ways": rng.choice([True, False]),
        "mind_map": False,
        "infographic": False,
        "examples": rng.choice([True, False]),
        "output_mode": rng.choice(output_modes),
        "student_mode": rng.choice([True, False]),
        "video_refs": False,
        "followup": False,
        "practice": False,
        "next_steps": False,
    }
    return {"context": context, "settings": chat_settings}


def _install_fakes(monkeypatch, record: dict):
    monkeypatch.setattr(app_settings, "CONTENT_GUARD_ENABLED", False)
    monkeypatch.setattr(AIService, "_get_gemini_async", lambda self: object())
    monkeypatch.setattr(AIService, "_get_openai", lambda self: None)

    async def fake_stream_gemini(self, system_prompt, turns, params, *a, **k):
        record["stream_gemini_system"] = system_prompt
        record["stream_gemini_turns"] = turns
        record["stream_params"] = params
        yield "ok"

    async def fake_stream_anthropic(self, system_prompt, turns, params, *a, **k):
        record["stream_claude_system"] = system_prompt
        record["stream_claude_messages"] = turns
        yield "ok"

    async def fake_gemini_retry(client, model, contents, config=None, max_retries=2, temperature=None):
        record["chat_gemini_temperature"] = temperature
        record["chat_gemini_system"] = getattr(config, "system_instruction", None)
        record["chat_gemini_turns"] = [
            {"role": c.role, "content": "".join(p.text for p in c.parts)} for c in contents
        ]
        return SimpleNamespace(text="ok")

    async def fake_create(**kwargs):
        record["chat_claude_system"] = kwargs["system"]
        record["chat_claude_messages"] = kwargs["messages"]
        return SimpleNamespace(content=[SimpleNamespace(text="ok")])

    fake_client = SimpleNamespace(messages=SimpleNamespace(create=fake_create))
    monkeypatch.setattr(AIService, "_stream_gemini", fake_stream_gemini)
    monkeypatch.setattr(AIService, "_stream_anthropic", fake_stream_anthropic)
    monkeypatch.setattr(ai_mod, "_gemini_generate_with_retry", fake_gemini_retry)
    monkeypatch.setattr(AIService, "_get_anthropic", lambda self: fake_client)


async def capture(monkeypatch, context: dict, chat_settings: dict) -> dict:
    """Run stream_chat and chat down both provider paths; return what each provider saw."""
    record: dict = {}
    _install_fakes(monkeypatch, record)
    ai = AIService()
    msgs = [dict(m) for m in MESSAGES]

    async for _ in ai.stream_chat(messages=msgs, context=dict(context), chat_settings=dict(chat_settings), has_files=False):
        pass
    async for _ in ai.stream_chat(messages=msgs, context=dict(context), chat_settings=dict(chat_settings), has_files=True):
        pass
    await ai.chat(messages=msgs, context=dict(context), chat_settings=dict(chat_settings), has_files=False)
    await ai.chat(messages=msgs, context=dict(context), chat_settings=dict(chat_settings), has_files=True)
    return record
