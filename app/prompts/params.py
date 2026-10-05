"""Generation parameters derived from the settings (plan §5, amendment 5).

max_tokens is a SAFETY CEILING sized from the length target — the prompt
controls length. It exists so long answers (deep-dive + 3-ways, Tamil/Hindi
scripts that cost more tokens per word) are never truncated.

temperature (first matching rule wins):
    sources attached (RAG)                                     → 0.3
    math/science subject, math-looking question, or expert     → 0.3
    friend / coach personality                                 → 0.7
    otherwise                                                  → 0.5
Whether a model receives `temperature` is decided by MODEL FAMILY
(`supports_temperature`), never by a hardcoded model name:
    Claude                      → never (current models reject it: "deprecated for this model")
    Gemini 3.x and later        → never (provider default, per Gemini 3 guidance)
    Gemini 2.x and earlier      → yes
    OpenAI reasoning (o-series, gpt-5*) → never (they reject it)
    other OpenAI chat models    → yes
"""
import re
from dataclasses import dataclass
from typing import Optional

from app.prompts.matrices import (
    MATH_TOKEN_FACTOR, MAX_TOKENS_CEILING, MAX_TOKENS_FLOOR, TOKEN_HEADROOM, TOKENS_PER_WORD,
    grade_band, target_words,
)
from app.schemas.ai_payloads import AIContextPayload, ChatSettingsPayload

THREE_WAYS_EXTRA_WORDS = 200          # the three labelled parts, upper bound
_FACTUAL_SUBJECT = re.compile(r"math|physics|chemistry|science|biology|statistics|accountancy|computer", re.I)
_MATH_SIGNAL = re.compile(r"\$|=|\d\s*[-+*/^×÷]\s*\d|[²³√∫∑π]|\b(solve|calculate|prove|derive|simplify|evaluate|factori[sz]e)\b", re.I)

_GEMINI_MAJOR = re.compile(r"gemini-(\d+)", re.I)
_OPENAI_REASONING = re.compile(r"^(o\d|gpt-5)", re.I)


def supports_temperature(provider: str, model: Optional[str]) -> bool:
    """Model-family check for the `temperature` parameter (see module docstring)."""
    if provider == "claude":
        return False
    if provider == "gemini":
        m = _GEMINI_MAJOR.search(model or "")
        return bool(m) and int(m.group(1)) < 3
    if provider == "openai":
        return not _OPENAI_REASONING.match(model or "")
    return False


@dataclass(frozen=True)
class GenParams:
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

    def temperature_for(self, provider: str, model: Optional[str]) -> Optional[float]:
        return self.temperature if supports_temperature(provider, model) else None


def is_math_heavy(question: Optional[str], subject: Optional[str]) -> bool:
    return bool((question and _MATH_SIGNAL.search(question)) or (subject and re.search(r"math", subject, re.I)))


def generation_params(
    ctx: Optional[AIContextPayload],
    settings: Optional[ChatSettingsPayload],
    question: Optional[str],
    has_files: bool,
    student_mode: bool,
) -> GenParams:
    """Params for an AI Assistant turn.

    Only the AI Assistant sends chat settings. Every other caller — internal
    generators, and context-passing features such as career analysis, Audio
    Q&A, Vault query and the Playground stream — gets GenParams() (no
    max_tokens, no temperature), i.e. its previous provider defaults
    (regression R2 in REGRESSION_CHECK.md).
    """
    if settings is None:
        return GenParams()
    language = (ctx.language if ctx else None) or "en"
    subject = ctx.subject if ctx else None
    length = settings.content_length if settings else "summary"
    band = grade_band(ctx.grade) if (ctx and student_mode and ctx.grade) else None

    _, words = target_words(band, length, language)
    if settings is not None and settings.explain_3ways:
        words += THREE_WAYS_EXTRA_WORDS
    factor = TOKENS_PER_WORD.get(language, 2.0)
    if is_math_heavy(question, subject):
        factor *= MATH_TOKEN_FACTOR
    max_tokens = int(min(MAX_TOKENS_CEILING, max(MAX_TOKENS_FLOOR, words * factor * TOKEN_HEADROOM)))

    if has_files:
        temperature = 0.3
    elif ((subject and _FACTUAL_SUBJECT.search(subject)) or is_math_heavy(question, subject)
          or (settings is not None and settings.difficulty == "expert")):
        temperature = 0.3
    elif settings is not None and settings.personality in ("friend", "coach"):
        temperature = 0.7
    else:
        temperature = 0.5
    return GenParams(max_tokens=max_tokens, temperature=temperature)
