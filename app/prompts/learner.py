"""<learner_profile> fragments: grade band, board overlay, subject (plan §3).

Phase 2 rewrite (audit P1, P4): band fragments carry vocabulary, sentence
length and scope only — no response-length or engagement rules; board
overlays say WHAT to emphasise, never how long. The previous long profiles are
archived in tests/golden/archive/prompts_v0.json.

Board registry (amendment 6): boards are recognised by case-insensitive
substring aliases, checked in registry order. A more specific regional board
(e.g. TN_STATE) is listed before the generic STATE entry it refines. To add a
state board, append a BoardSpec with its aliases and set `profile_key` to the
overlay it should render with (its own key once it has a dedicated overlay).
"""
import re
from dataclasses import dataclass
from typing import Optional

from app.prompts.matrices import grade_band


@dataclass(frozen=True)
class BoardSpec:
    key: str                       # stable registry key
    aliases: tuple[str, ...]       # upper-case substrings that identify the board
    profile_key: str               # which overlay text renders for it


BOARD_REGISTRY: tuple[BoardSpec, ...] = (
    # Amendment 6: Tamil Nadu State Board (Samacheer Kalvi). Other states: add a
    # BoardSpec here, plus a stage branch and an overlay below.
    BoardSpec("TN_STATE", ("TAMIL NADU", "SAMACHEER"), "TN_STATE"),
    BoardSpec("CBSE", ("CBSE",), "CBSE"),
    BoardSpec("ICSE", ("ICSE",), "ICSE"),
    BoardSpec("STATE", ("STATE",), "STATE"),
    BoardSpec("IB", ("IB",), "IB"),
    BoardSpec("CAMBRIDGE", ("CAMBRIDGE",), "CAMBRIDGE"),
)
_BY_KEY = {b.key: b for b in BOARD_REGISTRY}

# Every name a board/programme can appear under in rendered text (used by tests:
# these may appear only inside <learner_profile> and hard rule 2 — amendment 1).
BOARD_NAME_TERMS: tuple[str, ...] = (
    "CBSE", "NCERT", "ICSE", "ISC", "IB", "PYP", "MYP", "Diploma Programme", "Cambridge",
    "IGCSE", "AS & A Level", "Samacheer", "State board", "SSLC", "Higher Secondary",
    "Tamil Nadu", "HSC",
)

# Exam intent in the question (plan §3.3): exam notes even below detailed length.
EXAM_INTENT = re.compile(
    r"\b(exams?|board|marks?|important questions?|previous[- ]years?|pyqs?|question papers?|"
    r"sample papers?|blueprint|revision)\b",
    re.I,
)


def resolve_board_key(board: Optional[str]) -> Optional[str]:
    """Registry key for a free-text board label, or None if unrecognised."""
    if not board:
        return None
    upper = board.upper()
    for spec in BOARD_REGISTRY:
        if any(alias in upper for alias in spec.aliases):
            return spec.key
    return None


def board_profile_key(board_key: Optional[str]) -> Optional[str]:
    """Overlay key a registry board renders with."""
    return _BY_KEY[board_key].profile_key if board_key in _BY_KEY else None


# ── band fragments (plan §3.1) ───────────────────────────────────────────────

def band_fragment(grade: int) -> str:
    band = grade_band(grade)
    if band == "early-primary":
        age = grade + 5
        return (
            f"Learner: Grade {grade}, about {age} years old. Vocabulary: words a {age}-year-old uses; give "
            "any new word a simple meaning straight away. Sentences: short, about 8–12 words each, one idea "
            "per sentence. Scope: concrete, observable ideas; avoid symbols, formulas and abstract terms "
            "unless the question needs them, and then explain them with an everyday object (toys, food, "
            "animals, family)."
        )
    if band == "upper-primary":
        return (
            f"Learner: Grade {grade}. Vocabulary: clear everyday words; introduce subject terms with a "
            "plain-words meaning. Sentences: up to about 15 words each. Scope: grade-level concepts with "
            "simple cause and effect; simple formulas only in maths and only at grade level."
        )
    if band == "middle-school":
        return (
            f"Learner: Grade {grade}. Vocabulary: proper subject terminology, defined on first use. "
            "Sentences: up to about 20 words each. Scope: grade-level concepts with reasons and processes; "
            "algebraic and symbolic notation as taught at this grade."
        )
    return (
        f"Learner: Grade {grade}. Vocabulary: precise academic terminology. Sentences: up to about 25 "
        f"words each. Scope: the Grade {grade} syllabus; earlier-grade material may be assumed."
    )


# ── board stages and overlays (plan §3.2, amendments 3 and 10) ───────────────

def board_stage(profile_key: str, grade: Optional[int], board_label: str) -> str:
    """Curriculum stage named in the overlay (plan §3.2 table)."""
    if not grade:
        return {"CBSE": "CBSE (NCERT)", "ICSE": "ICSE / ISC", "STATE": board_label,
                "TN_STATE": "Tamil Nadu State Board (Samacheer Kalvi)",
                "IB": "International Baccalaureate", "CAMBRIDGE": "Cambridge International"}.get(profile_key, board_label)
    if profile_key == "CBSE":
        return f"NCERT Class {grade}" + (" (board-exam year)" if grade in (10, 12) else "")
    if profile_key == "ICSE":
        return f"ICSE Class {grade}" if grade <= 10 else f"ISC Class {grade}"
    if profile_key == "TN_STATE":
        if grade <= 10:
            return f"Tamil Nadu State Board (Samacheer Kalvi) Class {grade}" + (" (SSLC year)" if grade == 10 else "")
        return (f"Tamil Nadu Higher Secondary, {'First' if grade == 11 else 'Second'} Year "
                f"(+{grade - 10}{', HSC' if grade == 12 else ''})")
    if profile_key == "STATE":
        if grade <= 10:
            return f"State board Class {grade}" + (" (SSLC/SSC year)" if grade == 10 else "")
        return f"Higher Secondary, Class {grade} (+{grade - 10})"
    if profile_key == "IB":
        if grade <= 5:
            return f"IB Primary Years Programme (PYP), Grade {grade}"
        if grade <= 10:
            return f"IB Middle Years Programme (MYP) Year {grade - 5}"
        return f"IB Diploma Programme (DP) Year {grade - 10}"
    if profile_key == "CAMBRIDGE":
        if grade <= 6:
            return f"Cambridge Primary Stage {grade}"
        if grade <= 8:
            return f"Cambridge Lower Secondary Stage {grade}"
        if grade <= 10:
            return "Cambridge IGCSE"
        return "Cambridge International AS & A Level"
    return f"{board_label}, Class {grade}"


def board_overlay(profile_key: Optional[str], board_label: str, grade: Optional[int], content_length: str) -> str:
    """What to emphasise for the board — never how long (audit P1)."""
    if profile_key is None:
        stage = f"{board_label}, Class {grade}" if grade else board_label
        return (f"Curriculum: {stage}. Follow that board's terminology where known; otherwise standard "
                "national-curriculum conventions.")
    stage = board_stage(profile_key, grade, board_label)
    if profile_key == "CBSE":
        # Amendment 10: concept framing vs problem-solving framing.
        return (
            f"Curriculum: {stage}. Use NCERT terminology and give definitions in NCERT wording. Unless the "
            "user asks for a particular format, frame concept questions as the NCERT chapter does (definition → "
            "explanation → example) and problem-solving questions as given → formula → steps → answer with "
            "units. Examples from Indian daily life, geography and history."
        )
    if profile_key == "ICSE":
        return (
            f"Curriculum: {stage}. Precise, complete definitions; explain why as well as what, with cause and "
            "effect; in science, include experiment → observation → inference where relevant. A mix of "
            "Indian and international examples."
        )
    if profile_key == "TN_STATE":
        return (
            f"Curriculum: {stage}. Use the Samacheer Kalvi textbook terminology and definitions (in a Tamil "
            "answer, the Tamil textbook terms). Plain, direct textbook-style phrasing. Unless the user asks for "
            "a particular format, give one clear definition, then a short explanation, and solve problems as "
            "given → formula → steps → answer with units. Examples from everyday life in Tamil Nadu (home, "
            "farms, markets, festivals such as Pongal). Avoid idioms."
        )
    if profile_key == "STATE":
        return (
            f"Curriculum: {stage}. Plain, direct textbook-style phrasing; unless the user asks for a particular "
            "format, give one clear definition, then a short explanation. Familiar local examples (home, market, "
            "farm, festival). Avoid idioms."
        )
    if profile_key == "IB":
        # Amendment 3: guiding-question opening only for detailed / deep-dive.
        opening = (" Open with one short guiding question, then answer it."
                   if content_length in ("detailed", "deep-dive") else "")
        return (
            f"Programme: {stage}. Inquiry style: connect to real-world and global contexts and, where "
            f"natural, to other subjects. Use IB command terms (describe, explain, analyse, evaluate). "
            f"Diverse international examples.{opening}"
        )
    # CAMBRIDGE
    return (
        f"Curriculum: {stage}. Precise, formal wording that answers the command word (state, describe, "
        "explain, evaluate); show full working with units and sensible significant figures. International, "
        "data-supported examples."
    )


# Exam orientation (plan §3.3), Grade 9+ only. Mark patterns may be described
# but the grade/board must never be named in the answer (hard rule 2,
# amendment 1) — so the wording says "the textbook", not the board.
# Mark patterns are given only where well established; TN/State/ICSE stay
# generic because their current patterns vary by year and subject.
_EXAM_NOTES = {
    "CBSE": ("End with short exam notes: key terms in bold, the definition in the textbook's wording, one "
             "common mistake, and how it is typically asked (1/2/3/5-mark)."),
    "ICSE": ("End with short exam notes: precise definitions, reasoning-type points, a labelled-diagram "
             "description if relevant, and one common error."),
    "STATE": ("End with short exam notes: the definition first, then the short point-wise answer the exam "
              "expects, and one common mistake."),
    "TN_STATE": ("End with short exam notes: key terms in bold, the definition worded as in the textbook, one "
                 "common mistake, and the short- and long-answer forms it is typically asked in."),
    "IB": "End with a short note on which command term the answer satisfies and one assessment-criteria tip.",
    "CAMBRIDGE": "End with mark-scheme-style key points and one common examiner-reported error.",
}


def wants_exam_notes(profile_key: Optional[str], grade: Optional[int], content_length: str,
                     question: Optional[str]) -> bool:
    """Grade 9+, a known board, and (detailed/deep-dive length or exam intent in the question)."""
    if profile_key not in _EXAM_NOTES or not grade or grade < 9:
        return False
    if profile_key == "IB" and grade < 11:      # IB: Diploma Programme only
        return False
    return content_length in ("detailed", "deep-dive") or bool(question and EXAM_INTENT.search(question))


def subject_line(subject: str, stage: Optional[str]) -> str:
    """Subject alignment (plan §2.9); off-subject brevity is precedence item 3."""
    scope = f"the {stage} {subject} syllabus" if stage else f"the {subject} syllabus for this learner"
    return (f"Subject: {subject}. Align with {scope}. If the question is outside {subject}, still answer it "
            f"correctly but briefly, without forcing a link to {subject}.")


def learner_profile(
    grade: Optional[int],
    board: Optional[str],
    board_key: Optional[str],
    subject: Optional[str],
    content_length: str,
    question: Optional[str] = None,
) -> str:
    """Body of <learner_profile>; empty string when there is nothing learner-specific."""
    parts: list[str] = []
    if grade:
        parts.append(band_fragment(grade))
    profile_key = board_profile_key(board_key)
    stage = None
    if board:
        overlay = board_overlay(profile_key, board, grade, content_length)
        if wants_exam_notes(profile_key, grade, content_length, question):
            overlay += " " + _EXAM_NOTES[profile_key]
        parts.append(overlay)
        # Subject line names the stage without its exam-year note.
        stage = (re.sub(r" \((?:board-exam|SSLC/SSC|SSLC) year\)", "", board_stage(profile_key, grade, board))
                 if profile_key else None)
    if subject:
        parts.append(subject_line(subject, stage))
    return "\n".join(parts)
