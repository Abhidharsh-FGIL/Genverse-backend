"""<response_settings> fragments (plan §2.5–2.8; amendment 2).

Exactly one line per ACTIVE dimension (audit P6): disabled toggles emit
nothing. The length line is the only length authority (audit P1); difficulty
is relative to the learner's grade in student mode and absolute otherwise
(plan D4, audit G10); personality and engagement are a single fragment
(audit P4).
"""
from typing import Optional

from app.prompts.matrices import target_words

# ── length (plan §2.5) ───────────────────────────────────────────────────────

# Structure guidance per band × length; the word range comes from matrices.
_LENGTH_SHAPE: dict[str, dict[str, str]] = {
    "early-primary": {
        "small": "1–2 short sentences",
        "summary": "2 tiny paragraphs or 3–4 bullets, one idea per sentence",
        "detailed": "short paragraphs or First… Then… Finally… steps; no headings",
        "deep-dive": "a short story or step sequence with one simple activity to try; no formulas",
    },
    "upper-primary": {
        "small": "1–3 sentences",
        "summary": "a definition plus one explanation paragraph, or 3–5 bullets",
        "detailed": "2–3 mini-sections, numbered steps for any process",
        "deep-dive": "short sections, one worked example if the topic is computational, one real-life application",
    },
    "middle-school": {
        "small": "1–3 sentences",
        "summary": "a focused answer with key terms in bold",
        "detailed": "short headed sections with one worked example if the topic is computational",
        "deep-dive": "headed sections with worked examples, one common misconception and links to related topics",
    },
    "high-school": {
        "small": "1–3 sentences, including the key formula if there is one",
        "summary": "the key result with its reasoning",
        "detailed": "sections with a derivation or full worked solution where relevant",
        "deep-dive": "headed sections with derivations, edge cases and worked examples",
    },
    "off": {
        "small": "1–3 sentences",
        "summary": "a focused answer",
        "detailed": "short sections",
        "deep-dive": "headed sections covering the topic in depth",
    },
}


def length_line(band: Optional[str], content_length: str, language: str) -> str:
    lo, hi = target_words(band, content_length, language)
    shape = _LENGTH_SHAPE[band or "off"][content_length]
    if content_length == "small":
        return f"Length: at most {hi} words — {shape}. This is the only length rule."
    return f"Length: about {lo}–{hi} words — {shape}. This is the only length rule."


# ── difficulty (plan §2.6, amendment 2) ──────────────────────────────────────

_DIFFICULTY_BAND: dict[str, dict[str, str]] = {
    "early-primary": {
        "easy": "one core fact with an everyday comparison",
        "medium": "the fact plus a simple why",
        "hard": "two-step reasoning or a simple comparison",
        "expert": "what-if challenge thinking that uses only grade-level ideas, with no jargon",
    },
    "upper-primary": {
        "easy": "the core definition and one example",
        "medium": "the definition, how or why it happens, and an example",
        "hard": "applying the idea to a new situation in several steps",
        "expert": "extension puzzles that link chapters, with no formal notation beyond the grade",
    },
    "middle-school": {
        "easy": "the key idea in plain terms",
        "medium": "standard textbook depth",
        "hard": "multi-step problems and common misconceptions",
        "expert": "olympiad-style reasoning (NSO/IMO type) within the grade's concepts",
    },
    "high-school": {
        "easy": "the core result and the intuition behind it",
        "medium": "the standard expected in the final school exam",
        "hard": "higher-order and competitive-entry application",
        # expert is board-aware — see _HIGH_EXPERT
    },
}

# Amendment 2: high-band expert depends on the board. No board names are used
# here (amendment 1) and IB/Cambridge never mention JEE/NEET.
_HIGH_EXPERT = {
    "indian": "JEE-Advanced / NEET / olympiad rigour, with derivations and edge cases",
    "IB": "the depth that earns the top mark band, with extended-essay-style rigour and evaluation",
    "CAMBRIDGE": "A*-grade / olympiad level, with derivations and edge cases",
    "default": "olympiad-level rigour, with derivations and edge cases",
}

_LEVEL_POSITION = {"easy": "at its easy end", "medium": "at its typical level",
                   "hard": "at its upper end", "expert": "at its extension end"}

_DIFFICULTY_ABSOLUTE = {"easy": "a layperson", "medium": "an educated non-specialist",
                        "hard": "undergraduate", "expert": "specialist / graduate"}


def _high_expert(profile_key: Optional[str]) -> str:
    if profile_key in ("CBSE", "ICSE", "STATE", "TN_STATE"):
        return _HIGH_EXPERT["indian"]
    return _HIGH_EXPERT.get(profile_key or "default", _HIGH_EXPERT["default"])


def difficulty_line(level: str, grade: Optional[int], band: Optional[str], profile_key: Optional[str]) -> str:
    if not grade or not band:
        return f"Difficulty: {level} — pitch the answer at {_DIFFICULTY_ABSOLUTE[level]} level."
    if band == "high-school" and level == "expert":
        meaning = _high_expert(profile_key)
        tail = "Brief, clearly marked beyond-syllabus asides are allowed."
    else:
        meaning = _DIFFICULTY_BAND[band][level]
        tail = "Do not introduce content from later grades."
    return (f"Difficulty: {level} — stay within the Grade {grade} syllabus and work "
            f"{_LEVEL_POSITION[level]}: {meaning}. {tail}")


# ── personality × engagement (plan §2.7, audit P4) ───────────────────────────

_VOICE = {
    "mentor": "warm and encouraging; praise effort and correct mistakes gently (\"Almost — let's look at it this way\").",
    "coach": ("direct and action-oriented. For homework or problem-solving questions, give a short hint first, "
              "then the full worked solution; for factual questions, answer directly."),
    "professor": "formal, rigorous and clearly structured, with precise terminology.",
    "friend": "casual everyday language with light humour, still accurate.",
    "technical-expert": "precise technical terminology, mechanisms and edge cases.",
}
_ENDING = {
    "mentor": "End with one quick check question.",
    "coach": "End with one short challenge to try.",
    "professor": "End with a one-line key takeaway.",
    "friend": "End with one casual \"try this\" question.",
    "technical-expert": "End with a one-line key takeaway.",
}
_BAND_MODIFIER = {
    "early-primary": "Phrase any question playfully and simply.",
    "upper-primary": "Phrase any question playfully and simply.",
    "high-school": "Questions may be exam-style.",
}


def personality_line(personality: str, band: Optional[str], content_length: str, student_mode: bool) -> str:
    parts = [f"Personality: {_VOICE[personality]}"]
    if student_mode and content_length != "small":
        if band in _BAND_MODIFIER and personality not in ("professor", "technical-expert"):
            parts.append(_BAND_MODIFIER[band])
        parts.append(_ENDING[personality])
    return " ".join(parts)


# ── explain in 3 ways (plan §2.5, D5) and examples (plan §2.8, audit P2) ─────

THREE_WAYS_MARKER = "**Analogy:**"


def three_ways_line(content_length: str) -> str:
    size = "1–2 sentences" if content_length in ("small", "summary") else "2–4 sentences"
    return (f"Explain in 3 Ways: after the main answer, add a --- divider and three labelled parts — "
            f"{THREE_WAYS_MARKER}, **Technical:**, **Real-World Example:** — {size} each. These are in "
            f"addition to the length target.")


# D5: only when the toggle is off AND earlier answers in this chat used the format.
THREE_WAYS_OFF_AFTER_USE = ("Explain in 3 Ways is off for this answer: do not add Analogy / Technical / "
                            "Real-World Example sections.")

_EXAMPLES = {
    "small": "Examples: at most one short inline example.",
    "summary": "Examples: one concrete example.",
    "detailed": "Examples: two concrete examples.",
    "deep-dive": "Examples: two or three, including one real-world application.",
}


def examples_line(content_length: str) -> str:
    return _EXAMPLES[content_length]
