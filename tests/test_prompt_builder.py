"""Prompt builder: matrix, targeted and snapshot tests (plan §8.2, Phase 2).

Matrix: grade {2,5,8,11} × board {6} × language {en,hi,ta,ar} × student_mode {on,off}
(192 parametrised cases) × content_length {4} × difficulty {4} × provider {3} looped
inside each case = 9,216 rendered prompts. Pure functions — no network, no DB.

Snapshots: tests/snapshots/prompt_{A..F}.txt (plan §8.4 combos). Regenerate
deliberately with:  UPDATE_SNAPSHOTS=1 .venv/bin/python -m pytest tests/test_prompt_builder.py
"""
import itertools
import os
import re
from pathlib import Path

import pytest

from app.prompts.builder import (
    EARLIER_CONVERSATION, build_system_prompt, effective_student_mode, history_uses_three_ways,
    split_sections, to_provider_messages,
)
from app.prompts.language import language_reminder
from app.prompts.params import GenParams, generation_params
from app.prompts.learner import BOARD_NAME_TERMS
from app.prompts.settings import THREE_WAYS_MARKER, THREE_WAYS_OFF_AFTER_USE
from app.schemas.ai_payloads import normalize_context, normalize_settings

GRADES = (2, 5, 8, 11)
BOARDS = ("CBSE", "ICSE", "State Board", "IB", "Cambridge", "Tamil Nadu State Board")
INDIAN_BOARDS = ("CBSE", "ICSE", "State Board", "Tamil Nadu State Board")
EXAM_NOTE_MARKERS = ("exam notes", "command term the answer satisfies", "mark-scheme-style key points")
LANGS = ("en", "hi", "ta", "ar")
LENGTHS = ("small", "summary", "detailed", "deep-dive")
DIFFICULTIES = ("easy", "medium", "hard", "expert")
PROVIDERS = ("gemini", "claude", "openai")

# A response-LENGTH number: "120–180 words", "8 lines", "5 lines max" — but not a
# per-sentence size like "about 8–12 words each".
LENGTH_NUMBER = re.compile(
    r"\b\d+\s*(?:[-–]|to)\s*\d+\s*(?:words|lines)\b(?!\s+each)"
    r"|\b\d+\s+(?:words|lines)\b(?!\s+each)"
    r"|\blines?\s+max\b",
    re.I,
)
BOARD_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(t) for t in BOARD_NAME_TERMS) + r")(?![A-Za-z])")


def render(grade, board, lang, sm, length="summary", difficulty="medium", personality="mentor",
           subject=None, examples=False, three_ways=False, provider="gemini", history=False, question=None):
    ctx = normalize_context({"student_mode": sm, "grade": grade, "board": board,
                             "subject": subject, "language": lang})
    cs = normalize_settings({"student_mode": sm, "content_length": length, "difficulty": difficulty,
                             "personality": personality, "examples": examples, "explain_3ways": three_ways})
    return build_system_prompt(ctx, cs, provider, history_has_three_ways=history, question=question)


def _hard_rules_except_rule2(sections: dict) -> str:
    return "\n".join(l for l in sections["hard_rules"].splitlines() if not l.startswith("2. "))


# ── the matrix ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("grade, board, lang, sm", list(itertools.product(GRADES, BOARDS, LANGS, (True, False))))
def test_matrix(grade, board, lang, sm):
    for length, difficulty in itertools.product(LENGTHS, DIFFICULTIES):
        rendered = {p: render(grade, board, lang, sm, length, difficulty, provider=p) for p in PROVIDERS}
        # Same text for every provider (provider only changes message structure).
        assert len(set(rendered.values())) == 1
        prompt = rendered["gemini"]
        s = split_sections(prompt)
        label = f"g{grade} {board} {lang} sm={sm} {length} {difficulty}"

        # Each section exactly once; precedence + language always present.
        for tag in ("role", "hard_rules", "language", "response_settings", "precedence"):
            assert prompt.count(f"<{tag}>") == 1, f"{label}: <{tag}> count"

        # P1: exactly one length directive; no length numbers anywhere else.
        assert prompt.count("Length:") == 1 and prompt.count("This is the only length rule.") == 1, label
        length_line = next(l for l in s["response_settings"].splitlines() if l.startswith("Length:"))
        assert not LENGTH_NUMBER.search(prompt.replace(length_line, "")), f"{label}: stray length number"

        # P6 / D5: disabled toggles emit nothing; no "turned off" language.
        assert "Explain in 3 Ways" not in prompt and THREE_WAYS_MARKER not in prompt, label
        assert "Examples:" not in prompt, label
        assert not re.search(r"turned off|is off\b|currently off", prompt, re.I), label

        # Requested-format precedence: exactly once, after proportionality, before response settings.
        prec = s["precedence"]
        assert prec.count("Requested format:") == 1, label
        assert prec.index("3. Proportionality") < prec.index("4. Requested format") < prec.index("5. Response settings"), label
        # Board answer-framing is always conditional on no format being requested.
        if sm and re.search(r"definition → explanation|definition, then a short explanation|given → formula", s["learner_profile"]):
            assert "nless the user asks for a particular format" in s["learner_profile"], label

        # D1: the only emoji mention is the ban.
        assert prompt.lower().count("emoji") == 1 and "Do not use emojis." in prompt, label

        # Amendment 1: board names only in <learner_profile> or as hard rule 2's forbidden example.
        for tag in ("role", "language", "response_settings", "precedence"):
            assert not BOARD_RE.search(s[tag]), f"{label}: board name in <{tag}>"
        assert not BOARD_RE.search(_hard_rules_except_rule2(s)), label

        if sm:
            profile = s["learner_profile"]
            assert f"Learner: Grade {grade}" in profile, label
            assert re.search(r"^(Curriculum|Programme): ", profile, re.M), label
            # Ending question only when length allows it (P4).
            assert ("End with" in s["response_settings"]) == (length != "small"), label
            # Amendment 2: board-aware expert for the high band.
            if grade >= 10 and difficulty == "expert":
                if board in INDIAN_BOARDS:
                    assert "JEE-Advanced" in s["response_settings"], label
                else:
                    assert "JEE" not in prompt and "NEET" not in prompt, label
            # Amendment 3: IB guiding question only for detailed / deep-dive.
            if board == "IB":
                assert ("guiding question" in profile) == (length in ("detailed", "deep-dive")), label
            # Amendment 10: CBSE concept vs problem-solving framing.
            if board == "CBSE":
                assert "definition → explanation → example" in profile
                assert "given → formula → steps → answer with units" in profile
            # §3.3 exam notes: Grade 9+, detailed/deep-dive (no exam intent here), IB only DP.
            expect_notes = grade >= 9 and length in ("detailed", "deep-dive") and not (board == "IB" and grade < 11)
            assert any(m in profile for m in EXAM_NOTE_MARKERS) == expect_notes, label
            # D9 (amended): English term in parentheses only Grades 9–12, hi/ta/ar.
            assert ("you may add the English term in parentheses" in s["language"]) == (grade >= 9 and lang != "en"), label
        else:
            # G9: no learner profile, grade, board or subject when student mode is off.
            assert "<learner_profile>" not in prompt, label
            assert not re.search(r"Learner:|Curriculum:|Programme:|Subject:", prompt), label
            assert not re.search(r"Grade \d", prompt.replace(s["hard_rules"], "")), label
            assert not BOARD_RE.search(prompt.replace(s["hard_rules"], "")), label
            assert "End with" not in prompt, label
            assert "parentheses once" not in prompt, label
            assert "You are talking directly to a student" not in prompt, label


# ── targeted behaviour ───────────────────────────────────────────────────────

def test_internal_calls_get_role_and_hard_rules_only():
    s = split_sections(build_system_prompt(None, None))
    assert set(s) == {"role", "hard_rules"}


def test_caller_without_student_mode_or_grade_gets_language_but_no_profile():
    """Addition 4: audio Q&A / vault query / career send no student_mode or grade."""
    for raw in ({"language": "hi"}, {"stream": "Science", "language": "ta"}, {"grade": 8, "language": "en"}):
        ctx = normalize_context(raw)
        s = split_sections(build_system_prompt(ctx, None))
        assert set(s) == {"role", "hard_rules", "language", "precedence"}, raw
        assert not effective_student_mode(ctx, None)


def test_context_student_mode_counts_only_when_sent():
    assert effective_student_mode(normalize_context({"student_mode": True, "grade": 5}), None)
    assert not effective_student_mode(normalize_context({"grade": 5}), None)
    # chat settings always win (I7)
    assert not effective_student_mode(normalize_context({"student_mode": True}), normalize_settings({"student_mode": False}))


def test_unknown_language_gets_english_instruction():
    s = split_sections(render(8, "CBSE", "klingon", True))
    assert s["language"].startswith("Respond entirely in English")


def test_missing_language_gets_english_instruction():
    ctx = normalize_context({"student_mode": True, "grade": 8})
    assert split_sections(build_system_prompt(ctx, None))["language"].startswith("Respond entirely in English")


@pytest.mark.parametrize("length, size", [("small", "1–2 sentences"), ("summary", "1–2 sentences"),
                                          ("detailed", "2–4 sentences"), ("deep-dive", "2–4 sentences")])
def test_three_ways_on(length, size):
    prompt = render(5, "CBSE", "en", True, length, three_ways=True)
    assert THREE_WAYS_MARKER in prompt and f"{size} each" in prompt


def test_three_ways_off_after_earlier_use_gets_one_short_line():
    prompt = render(5, "CBSE", "en", True, history=True)
    assert THREE_WAYS_OFF_AFTER_USE in prompt
    assert prompt.count("Explain in 3 Ways") == 1


def test_history_marker_detection():
    assert history_uses_three_ways([{"role": "assistant", "content": f"x\n---\n{THREE_WAYS_MARKER} y"}])
    assert not history_uses_three_ways([{"role": "user", "content": THREE_WAYS_MARKER}])


@pytest.mark.parametrize("length, text", [("small", "at most one short inline example"),
                                          ("summary", "one concrete example"),
                                          ("detailed", "two concrete examples"),
                                          ("deep-dive", "two or three, including one real-world application")])
def test_examples_scale_with_length(length, text):
    assert text in render(8, "CBSE", "en", True, length, examples=True)


@pytest.mark.parametrize("subject, allowed", [(None, True), ("Physics", True), ("Mathematics", True),
                                              ("History", False), ("English Literature", False)])
def test_d9_parenthetical_only_for_science_maths(subject, allowed):
    lang_section = split_sections(render(11, "CBSE", "ta", True, subject=subject))["language"]
    assert ("you may add the English term in parentheses" in lang_section) == allowed


@pytest.mark.parametrize("label", ["Tamil Nadu State Board", "Samacheer Kalvi", "TN State Board (Tamil Nadu)"])
def test_tamil_nadu_overlay(label):
    profile = split_sections(render(10, label, "ta", True))["learner_profile"]
    assert "Curriculum: Tamil Nadu State Board (Samacheer Kalvi) Class 10 (SSLC year)." in profile
    assert "Samacheer Kalvi textbook terminology" in profile and "Pongal" in profile


@pytest.mark.parametrize("grade, stage", [(11, "Tamil Nadu Higher Secondary, First Year (+1)"),
                                          (12, "Tamil Nadu Higher Secondary, Second Year (+2, HSC)")])
def test_tamil_nadu_higher_secondary_stages(grade, stage):
    assert f"Curriculum: {stage}." in split_sections(render(grade, "Tamil Nadu State Board", "ta", True))["learner_profile"]


def test_subject_line_drops_exam_year_note():
    profile = split_sections(render(10, "Tamil Nadu State Board", "ta", True, subject="Science"))["learner_profile"]
    assert "Align with the Tamil Nadu State Board (Samacheer Kalvi) Class 10 Science syllabus." in profile


@pytest.mark.parametrize("question, expected", [
    ("important questions on light reflection", True),
    ("What will come in the board exam from this chapter?", True),
    ("give me 5 mark answers", True),
    ("what is light reflection", False),
])
def test_exam_intent_triggers_exam_notes_below_detailed(question, expected):
    profile = split_sections(render(10, "CBSE", "en", True, length="summary", question=question))["learner_profile"]
    assert ("exam notes" in profile) == expected


def test_no_exam_notes_below_grade_9_even_with_exam_intent():
    profile = split_sections(render(8, "CBSE", "en", True, length="deep-dive", question="important questions"))["learner_profile"]
    assert "exam notes" not in profile


def test_exam_notes_never_name_the_board():
    """Amendment 1: exam notes describe mark patterns, never name grade or board."""
    for board in ("CBSE", "ICSE", "State Board", "Tamil Nadu State Board", "Cambridge", "IB"):
        profile = split_sections(render(11, board, "en", True, length="detailed"))["learner_profile"]
        notes = profile[profile.index("End with"):].split("\n")[0]
        assert not BOARD_RE.search(notes), (board, notes)


def test_unrecognised_board_gets_generic_overlay():
    profile = split_sections(render(7, "Kerala Board", "en", True))["learner_profile"]
    assert "Curriculum: Kerala Board, Class 7." in profile


def test_subject_line_and_off_subject_rule():
    s = split_sections(render(8, "CBSE", "en", True, subject="Mathematics"))
    assert "Subject: Mathematics. Align with the NCERT Class 8 Mathematics syllabus." in s["learner_profile"]
    assert "outside the stated subject gets a correct but brief answer" in s["precedence"]


# ── turn normalisation (G14) and language reminder (P5) ──────────────────────

def test_to_provider_messages_merges_and_reminds_without_mutating():
    history = [{"role": "assistant", "content": "a"}, {"role": "system", "content": "s"},
               {"role": "user", "content": "u1"}, {"role": "user", "content": "u2"},
               {"role": "assistant", "content": "a2"}, {"role": "assistant", "content": "a3"}]
    before = [dict(m) for m in history]
    turns = to_provider_messages(history, "hi")
    assert history == before
    assert turns == [
        {"role": "user", "content": EARLIER_CONVERSATION},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": f"u1\n\nu2\n\n{language_reminder('hi')}"},
        {"role": "assistant", "content": "a2\n\na3"},
    ]


def test_no_reminder_when_language_is_none():
    assert to_provider_messages([{"role": "user", "content": "x"}], None) == [{"role": "user", "content": "x"}]


@pytest.mark.parametrize("lang, text", [("en", "(Answer in English.)"),
                                        ("ta", "(Answer only in Tamil, using Tamil script; keep math in standard LaTeX.)"),
                                        ("xx", "(Answer in English.)")])
def test_language_reminder_text(lang, text):
    assert language_reminder(lang) == text


# ── generation params (plan §5, amendment 5) ─────────────────────────────────

def _params(question="What is light?", has_files=False, **kw):
    ctx = normalize_context({"student_mode": kw.get("sm", True), "grade": kw.get("grade", 8), "board": "CBSE",
                             "subject": kw.get("subject"), "language": kw.get("lang", "en")})
    cs = normalize_settings({"student_mode": kw.get("sm", True), "content_length": kw.get("length", "summary"),
                             "difficulty": kw.get("difficulty", "medium"), "personality": kw.get("personality", "mentor"),
                             "explain_3ways": kw.get("three_ways", False)})
    return generation_params(ctx, cs, question, has_files, kw.get("sm", True))


@pytest.mark.parametrize("kw, temperature", [
    (dict(has_files=True, personality="friend"), 0.3),
    (dict(subject="Physics", personality="friend"), 0.3),
    (dict(question="Solve x^2 - 5x + 6 = 0", personality="coach"), 0.3),
    (dict(difficulty="expert", personality="friend"), 0.3),
    (dict(personality="friend"), 0.7),
    (dict(personality="coach"), 0.7),
    (dict(personality="professor"), 0.5),
])
def test_temperature_rules(kw, temperature):
    assert _params(**kw).temperature == temperature


@pytest.mark.parametrize("provider, model, sent", [
    ("gemini", "gemini-3.5-flash", False), ("gemini", "gemini-3-pro-preview", False),
    ("gemini", "gemini-4.0-ultra", False), ("gemini", "gemini-2.5-flash-lite", True),
    ("gemini", "gemini-2.0-flash", True), ("openai", "gpt-4o-mini", True), ("openai", "gpt-4.1", True),
    ("openai", "o3-mini", False), ("openai", "gpt-5", False), ("claude", "claude-sonnet-5-5", False),
    ("claude", "claude-sonnet-4-20250514", False), ("gemini", None, False),
])
def test_temperature_by_model_family(provider, model, sent):
    """Decision 2: temperature is a model-FAMILY check, never a hardcoded name."""
    p = _params(personality="friend")
    assert (p.temperature_for(provider, model) == 0.7) == sent
    assert (p.temperature_for(provider, model) is None) == (not sent)


def test_max_tokens_scales_and_has_headroom():
    small = _params(length="small").max_tokens
    deep_en = _params(grade=11, length="deep-dive").max_tokens
    deep_ta = _params(grade=11, length="deep-dive", lang="ta").max_tokens
    assert small == 512                                   # floor
    assert deep_en == int(1100 * 2.0 * 1.6)              # 3520
    assert deep_ta > 4096                                 # the old Claude cap would truncate Tamil deep-dives
    assert _params(grade=11, length="deep-dive", lang="ta", three_ways=True, question="solve x+1=2").max_tokens == 8192


def test_internal_calls_get_default_params():
    assert generation_params(None, None, "x", False, False) == GenParams()


def test_student_mode_off_difficulty_is_absolute():
    s = split_sections(render(8, "CBSE", "en", False, difficulty="hard"))
    assert s["response_settings"].splitlines()[1] == "Difficulty: hard — pitch the answer at undergraduate level."


# ── snapshots (plan §8.4) ────────────────────────────────────────────────────

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_COMBOS = {
    "A": dict(grade=2, board="State Board", lang="en", sm=True, length="small", difficulty="easy", personality="mentor"),
    "B": dict(grade=5, board="CBSE", lang="hi", sm=True, length="summary", difficulty="medium", personality="friend",
              three_ways=True),
    "C": dict(grade=8, board="ICSE", lang="ta", sm=True, length="detailed", difficulty="hard", personality="coach",
              subject="Science", examples=True),
    "D": dict(grade=11, board="CBSE", lang="en", sm=True, length="deep-dive", difficulty="expert",
              personality="professor", subject="Mathematics"),
    "E": dict(grade=11, board="IB", lang="ar", sm=True, length="detailed", difficulty="medium", personality="mentor"),
    "F": dict(grade=8, board="CBSE", lang="en", sm=False, length="summary", difficulty="hard",
              personality="technical-expert"),
    "G": dict(grade=10, board="Tamil Nadu State Board", lang="ta", sm=True, length="detailed", difficulty="medium",
              personality="mentor", subject="Science", question="important questions on light reflection"),
}


@pytest.mark.parametrize("name", sorted(SNAPSHOT_COMBOS))
def test_snapshot(name):
    prompt = render(**SNAPSHOT_COMBOS[name])
    path = SNAPSHOT_DIR / f"prompt_{name}.txt"
    if os.environ.get("UPDATE_SNAPSHOTS") == "1" or not path.exists():
        SNAPSHOT_DIR.mkdir(exist_ok=True)
        path.write_text(prompt)
    assert prompt == path.read_text(), f"prompt {name} changed — review, then UPDATE_SNAPSHOTS=1"
