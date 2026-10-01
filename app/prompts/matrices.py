"""Shared tailoring tables (plan §2.5, amendment 4).

Single source for the length targets used by BOTH the prompt builder (Phase 2)
and scripts/eval_tailoring.py, so the eval measures exactly what the prompt
asks for.
"""
from typing import Optional

BANDS = ("early-primary", "upper-primary", "middle-school", "high-school")
OFF = "off"  # student mode OFF → no grade band


def grade_band(grade: Optional[int]) -> Optional[str]:
    """Grades 1-3 early, 4-6 upper, 7-9 middle, 10-12 high (same cut-offs as before)."""
    if not grade:
        return None
    if grade <= 3:
        return "early-primary"
    if grade <= 6:
        return "upper-primary"
    if grade <= 9:
        return "middle-school"
    return "high-school"


# English word targets: band (or OFF) × content_length → (min_words, max_words).
LENGTH_WORDS: dict[str, dict[str, tuple[int, int]]] = {
    "early-primary": {"small": (5, 30), "summary": (50, 90), "detailed": (100, 160), "deep-dive": (180, 260)},
    "upper-primary": {"small": (8, 45), "summary": (90, 140), "detailed": (180, 280), "deep-dive": (300, 450)},
    "middle-school": {"small": (10, 60), "summary": (120, 180), "detailed": (280, 420), "deep-dive": (500, 750)},
    "high-school":   {"small": (10, 70), "summary": (150, 220), "detailed": (380, 550), "deep-dive": (700, 1100)},
    OFF:             {"small": (10, 60), "summary": (120, 200), "detailed": (300, 500), "deep-dive": (700, 1200)},
}

# Amendment 4: words needed for equivalent content differ by language
# (Tamil agglutinates → fewer, longer words). Starting values; calibrated from
# the first eval run (see eval_results/*/REPORT.md, "language calibration").
LANGUAGE_WORD_MULTIPLIER: dict[str, float] = {"en": 1.0, "hi": 1.1, "ar": 0.9, "ta": 0.75}

# Eval tolerance around the target band (amendment 4).
LENGTH_TOLERANCE = 0.25

# Greetings / small talk: 1–2 sentences whatever the length setting (precedence 3).
GREETING_MAX_SENTENCES = 2


def target_words(band: Optional[str], content_length: str, language: Optional[str]) -> tuple[int, int]:
    """Word target for a band (None → student mode OFF), length and language."""
    row = LENGTH_WORDS[band or OFF]
    lo, hi = row.get(content_length, row["summary"])
    m = LANGUAGE_WORD_MULTIPLIER.get((language or "en").lower(), 1.0)
    return round(lo * m), round(hi * m)


# Output tokens per prose word, by language (amendment 5). Starting values from
# plan §5, checked against the gpt-4o-mini baseline (en mean 1.74 / max 2.41,
# hi 1.97, ta 3.2 / max 3.97). RE-CALIBRATE from the first eval run on the
# restored primary model (eval REPORT.md, "Measured output tokens per word").
TOKENS_PER_WORD: dict[str, float] = {
    "en": 2.0, "fr": 2.0, "de": 2.0, "es": 2.0, "pt": 2.0,
    "hi": 4.5, "ta": 4.5, "te": 4.5, "ar": 4.5, "zh": 2.5,
}
# Math-heavy answers spend extra tokens on LaTeX (amendment 5).
MATH_TOKEN_FACTOR = 1.3
# max_tokens = words × tokens/word × HEADROOM: a safety ceiling, not a length control.
TOKEN_HEADROOM = 1.6
MAX_TOKENS_FLOOR, MAX_TOKENS_CEILING = 512, 8192
