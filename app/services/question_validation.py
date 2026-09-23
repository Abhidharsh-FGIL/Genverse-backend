r"""The ONE validation gate for assessment question content.

Every path that persists questions must run them through `validate_and_flag`.
Before this existed the checks lived inside AIService.finalize_generated_questions,
which only the two SSE/Celery generation paths called — so POST
/assessments/generate (the Daily Challenge), POST /assessments (save) and PATCH
/assessments/{id} all wrote straight to the database with no repair and no
validation at all. Anything those endpoints stored was unvalidated by
construction, which is exactly the kind of bypass that lets corrupted content
reach users.

This module deliberately does NOT reshape questions. It repairs the text of
whatever fields are present, under whichever key spelling they use
(correct_answer vs correctAnswer, marks vs points), and leaves structure alone —
because it runs on already-saved and human-edited content as well as on fresh
generations, where changing shape would break callers (the Daily Challenge quiz
reads correct_answer/marks straight out of question_json).

Checks applied, in the order they can catch things:
  1. Control characters — corruption with no Node dependency, so it still works
     when real KaTeX validation is unavailable.
  2. Structural — unbalanced $, unbalanced braces, bare chemical formulas.
  3. Over-escaped backslashes that survived repair.
  4. A real KaTeX parse of every math segment (skipped, loudly, if unavailable).
"""
from __future__ import annotations

import logging

from app.services.latex_validator import (
    repair_common_latex_issues, validate_questions_latex, option_text,
    has_over_escaped_command, has_control_characters, describe_control_characters,
)

log = logging.getLogger(__name__)

TEXT_FIELDS = ("text", "question", "correct_answer", "correctAnswer", "explanation")


def _repair_in_place(q: dict) -> None:
    """Repair every text field of `q`, preserving its existing shape/keys."""
    for key in TEXT_FIELDS:
        value = q.get(key)
        if isinstance(value, str) and value:
            q[key] = repair_common_latex_issues(value)

    opts = q.get("options")
    if isinstance(opts, list):
        new_opts = []
        for opt in opts:
            if isinstance(opt, str):
                new_opts.append(repair_common_latex_issues(opt))
            elif isinstance(opt, dict) and isinstance(opt.get("text"), str):
                new_opt = dict(opt)
                new_opt["text"] = repair_common_latex_issues(opt["text"])
                new_opts.append(new_opt)
            else:
                new_opts.append(opt)
        q["options"] = new_opts

    pairs = q.get("pairs")
    if isinstance(pairs, list):
        new_pairs = []
        for pair in pairs:
            if isinstance(pair, dict):
                new_pair = dict(pair)
                for side in ("left", "right"):
                    if isinstance(pair.get(side), str):
                        new_pair[side] = repair_common_latex_issues(pair[side])
                new_pairs.append(new_pair)
            else:
                new_pairs.append(pair)
        q["pairs"] = new_pairs


def _all_texts(q: dict) -> list[tuple[str, str]]:
    """[(field_label, text)] for every text-bearing field of `q`."""
    out: list[tuple[str, str]] = []
    for key in TEXT_FIELDS:
        value = q.get(key)
        if isinstance(value, str) and value:
            out.append((key, value))
    for i, opt in enumerate(q.get("options") or []):
        text = opt if isinstance(opt, str) else option_text(opt)
        if text:
            out.append((f"options[{i}]", text))
    for i, pair in enumerate(q.get("pairs") or []):
        if isinstance(pair, dict):
            for side in ("left", "right"):
                if isinstance(pair.get(side), str) and pair[side]:
                    out.append((f"pairs[{i}].{side}", pair[side]))
    return out


async def validate_and_flag(
    questions: list, *, repair: bool = True, context: str = "unknown",
) -> tuple[list, dict]:
    """Repair, validate and flag `questions` in place.

    Returns (questions, stats) where stats carries everything a caller needs to
    log a meaningful one-line summary:
        questions, flagged, katex_active, control_char_questions
    """
    from app.services.katex_check import check_segments, status as katex_status

    usable = [q for q in questions if isinstance(q, dict)]

    if repair:
        for q in usable:
            _repair_in_place(q)

    # Normalise to the snake_case spelling the shared validator expects, without
    # mutating the question: a question saved from the edit screen carries
    # correctAnswer, one straight from the generator carries correct_answer, and
    # a validator that knew only one spelling would silently skip half the
    # content it is supposed to be checking.
    normalised = []
    for q in usable:
        view = dict(q)
        if "correct_answer" not in view and "correctAnswer" in view:
            view["correct_answer"] = view["correctAnswer"]
        view.setdefault("id", "")
        normalised.append(view)

    issues: dict[str, list[str]] = await validate_questions_latex(normalised)

    # Real KaTeX parse of every math segment.
    from app.services.latex_validator import extract_math_segments
    seg_owners: dict[str, list[tuple[str, str]]] = {}
    for q in normalised:
        qid = str(q.get("id") or "")
        for field, text in _all_texts(q):
            for seg in extract_math_segments(text):
                seg_owners.setdefault(seg, []).append((qid, field))

    katex_errors = await check_segments(list(seg_owners))
    katex_active = katex_errors is not None
    if katex_errors:
        for seg, msg in katex_errors.items():
            for qid, field in seg_owners.get(seg, []):
                issues.setdefault(qid, []).append(
                    f"KaTeX parse error in {field}: {msg} (segment: {seg!r})"
                )

    # Over-escaped backslashes that survived repair.
    control_char_questions = 0
    for q in normalised:
        qid = str(q.get("id") or "")
        has_control = False
        for field, text in _all_texts(q):
            if has_over_escaped_command(text):
                issues.setdefault(qid, []).append(
                    f"over-escaped LaTeX backslash survived repair in {field}"
                )
            if has_control_characters(text):
                has_control = True
        if has_control:
            control_char_questions += 1

    flagged = {qid: msgs for qid, msgs in issues.items() if msgs}
    for q in usable:
        msgs = flagged.get(str(q.get("id") or ""))
        if msgs:
            q["needs_review"] = True
            q["review_reason"] = "; ".join(msgs)[:500]

    stats = {
        "questions": len(usable),
        "flagged": len(flagged),
        "katex_active": katex_active,
        "control_char_questions": control_char_questions,
    }

    # One line per validation pass, so "was this actually validated?" is
    # answerable from logs instead of guessed at.
    if not katex_active:
        log.warning(
            "[QuestionValidation] %s: validated %d question(s) WITHOUT real KaTeX "
            "parsing (%s). Only structural checks ran, so a question whose math "
            "fails to parse can still be saved. flagged=%d",
            context, stats["questions"], katex_status().get("katex_validation_detail"),
            stats["flagged"],
        )
    else:
        log.info(
            "[QuestionValidation] %s: validated %d question(s) with real KaTeX; "
            "flagged=%d control_char=%d",
            context, stats["questions"], stats["flagged"], control_char_questions,
        )
    if control_char_questions:
        log.error(
            "[QuestionValidation] %s: %d question(s) contain control characters — "
            "content was corrupted and text may have been LOST; these need "
            "regeneration, not repair.", context, control_char_questions,
        )
    for qid, msgs in flagged.items():
        log.warning("[QuestionValidation] %s: question %s FLAGGED: %s",
                    context, qid, "; ".join(msgs)[:300])

    return questions, stats
