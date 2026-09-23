r"""Regression guard: no stored assessment question may contain an over-escaped
LaTeX backslash.

This is the invariant the Assessment Hub corruption violated — a doubled
backslash immediately before a command letter, a spacing-macro comma or a
script marker, inside a math span. Genuine "\\" line breaks (matrix/cases/
aligned row separators, and "\\" before whitespace) are explicitly NOT
violations; see tests/test_latex_escaping.py for that boundary.

Skips cleanly when no database is reachable, so it is safe in CI without one.
Point it at any environment with DATABASE_URL / the standard DB_* settings.
"""
import asyncio
import json

import pytest

from app.services.latex_validator import has_over_escaped_command, option_text

pytestmark = pytest.mark.asyncio


FIELDS = ("text", "question", "correct_answer", "correctAnswer", "explanation")


def _offending_fields(question: dict) -> list[str]:
    """Names of this question's fields that still carry the corruption."""
    if not isinstance(question, dict):
        return []
    bad = []
    for key in FIELDS:
        value = question.get(key)
        if isinstance(value, str) and has_over_escaped_command(value):
            bad.append(key)
    for i, opt in enumerate(question.get("options") or []):
        if has_over_escaped_command(option_text(opt)):
            bad.append(f"options[{i}]")
    for i, pair in enumerate(question.get("pairs") or []):
        if isinstance(pair, dict):
            for side in ("left", "right"):
                if has_over_escaped_command(str(pair.get(side) or "")):
                    bad.append(f"pairs[{i}].{side}")
    return bad


async def _load_rows():
    try:
        from sqlalchemy import select
        from app.database import AsyncSessionLocal
        from app.models.assessment import PracticeAssessment
    except Exception as e:  # pragma: no cover - import-time config problems
        pytest.skip(f"database layer unavailable: {e}")

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(
                    PracticeAssessment.id,
                    PracticeAssessment.title,
                    PracticeAssessment.question_json,
                )
            )
            return result.all()
    except Exception as e:
        pytest.skip(f"no database reachable: {type(e).__name__}: {e}")


async def test_no_stored_question_has_over_escaped_latex():
    rows = await _load_rows()
    if not rows:
        pytest.skip("no assessments stored in this environment")

    violations = []
    for assessment_id, title, question_json in rows:
        if isinstance(question_json, str):
            try:
                question_json = json.loads(question_json)
            except json.JSONDecodeError:
                continue
        if not isinstance(question_json, list):
            continue
        for question in question_json:
            for field in _offending_fields(question):
                violations.append(
                    f"  assessment {assessment_id} ({title!r}) "
                    f"question {question.get('id')!r} field {field}"
                )

    assert not violations, (
        f"{len(violations)} stored question field(s) still contain an "
        f"over-escaped LaTeX backslash (\\\\ before a letter, comma or ^).\n"
        "Run scripts/migrate_collapse_over_escaped_latex.py to repair them.\n"
        + "\n".join(violations[:40])
    )
