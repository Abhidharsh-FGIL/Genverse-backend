"""<language> section and the language/script registries (plan §2.3, §4).

Phase 2: one authoritative language section replaces the old RESPONSE
LANGUAGE clause. Phase 3: the Gemini-only trailing [SYSTEM: ...] tag is replaced
by `language_reminder`, appended to the last user turn for every provider (P5).

AIService keeps `_LANGUAGE_NAMES` / `_SCRIPT_MAP` as aliases of the dicts
below (other features use them).
"""
import re
from typing import Optional

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Chinese (中文)",
    "ar": "Arabic (عربي)",
    "pt": "Portuguese",
}

# Maps language name → its writing script (used for strong script-enforcement instructions)
SCRIPT_MAP: dict[str, str] = {
    "hindi": "Devanagari (हिन्दी)",
    "tamil": "Tamil (தமிழ்)",
    "telugu": "Telugu (తెలుగు)",
    "kannada": "Kannada (ಕನ್ನಡ)",
    "malayalam": "Malayalam (മലയാളം)",
    "bengali": "Bengali (বাংলা)",
    "marathi": "Devanagari (मराठी)",
    "gujarati": "Gujarati (ગુજરાતી)",
    "punjabi": "Gurmukhi (ਪੰਜਾਬੀ)",
    "urdu": "Urdu (اردو)",
    "arabic": "Arabic (العربية)",
    "chinese": "Chinese characters (中文)",
}

# Plain names / scripts for the <language> section.
_PLAIN_NAME = {"en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "fr": "French",
               "de": "German", "es": "Spanish", "zh": "Chinese", "ar": "Arabic", "pt": "Portuguese"}
_SCRIPT = {"hi": "Devanagari script", "ta": "Tamil script", "te": "Telugu script", "ar": "Arabic script",
           "zh": "Chinese characters", "fr": "Latin script", "de": "Latin script", "es": "Latin script",
           "pt": "Latin script"}

# D9 (amended): English term in parentheses once, Grades 9–12, Science/Maths, hi/ta/ar only.
_PARENS_LANGS = {"hi", "ta", "ar"}
_SCIENCE_MATH = re.compile(r"math|science|physics|chemistry|biology|statistics|computer", re.I)

_BAND_VOCAB = {
    "early-primary": "Use everyday {L} words a child uses at home.",
    "upper-primary": "Use everyday {L} words a child uses at home.",
    "middle-school": "Use standard school-textbook {L}.",
    "high-school": "Use standard academic {L} terminology.",
}


def language_name(language: str) -> str:
    return LANGUAGE_NAMES.get(language.lower(), language)


def allows_english_parenthetical(language: str, grade: Optional[int], subject: Optional[str]) -> bool:
    """D9 (amended): Grades 9–12, hi/ta/ar, and a Science/Maths subject (or no subject set)."""
    if language not in _PARENS_LANGS or not grade or grade < 9:
        return False
    return subject is None or bool(_SCIENCE_MATH.search(subject))


def language_section(
    language: str,
    band: Optional[str] = None,
    grade: Optional[int] = None,
    subject: Optional[str] = None,
) -> str:
    """The single authoritative language instruction.

    `band` / `grade` / `subject` are passed only in student mode; they add the
    vocabulary-tier line and decide the D9 parenthetical exception.
    """
    lang = language if language in _PLAIN_NAME else "en"
    name = _PLAIN_NAME[lang]
    if lang == "en":
        lines = ["Respond entirely in English, even if the topic or earlier messages use another language."]
    else:
        script = _SCRIPT[lang]
        if allows_english_parenthetical(lang, grade, subject):
            parens = ("For a technical Science or Maths term only, you may add the English term in "
                      "parentheses once, on its first use; never do this for everyday words.")
        else:
            parens = "Do not add English translations in parentheses."
        lines = [
            f"Respond entirely in {name}, using only {script}. Do not mix in characters from other scripts. "
            f"Use the terms found in {name}-medium school textbooks; if no standard {name} term exists, "
            f"write the English term as-is. {parens} Keep LaTeX math, $\\ce{{}}$ formulas, code, variable "
            f"names and scientific symbols in standard Latin form. Use correct {name} spelling and "
            f"grammar — never transliterated English."
        ]
    if band in _BAND_VOCAB:
        lines.append(_BAND_VOCAB[band].format(L=name))
    return " ".join(lines)


def language_reminder(language: Optional[str]) -> str:
    """One-line reminder appended to the LAST user turn for every provider (audit P5).

    Built at provider-message time only: it is never saved to chat history and
    never reaches the content guard or the summary-intent regex (amendment 9).
    """
    lang = language if language in _PLAIN_NAME else "en"
    if lang == "en":
        return "(Answer in English.)"
    return f"(Answer only in {_PLAIN_NAME[lang]}, using {_SCRIPT[lang]}; keep math in standard LaTeX.)"
