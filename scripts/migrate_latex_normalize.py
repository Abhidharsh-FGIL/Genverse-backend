"""
One-off migration: repair and validate LaTeX in existing PracticeAssessment
rows (practice_assessments.question_json) — the source data behind the /u/
assessments question preview, Take view, and review screens.

Applies the same conservative, safe repairs the generation pipeline now runs
on new questions (app/services/latex_validator.repair_common_latex_issues):
unbalanced leading $ on a short option, double-escaped \\ce macros, and a
stray \\% outside math. Then re-validates every question (balanced $/$$,
balanced braces within each math segment, and bare chemical formulas that
should be wrapped in \\ce{...}) and reports which question IDs still fail —
those need manual review or regeneration, this script will not guess at them.

DRY RUN BY DEFAULT — prints a diff summary and the still-failing list, writes
nothing. Pass --apply to persist the repaired question_json/updated_at.

Run from backend directory:
    python -m scripts.migrate_latex_normalize            # dry run
    python -m scripts.migrate_latex_normalize --apply     # write changes
    python -m scripts.migrate_latex_normalize --limit 50  # sample first N rows
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import asyncio
import copy
import json

from sqlalchemy import select

from app.database import AsyncSessionLocal, engine
from app.models.assessment import PracticeAssessment
from app.services.latex_validator import (
    repair_common_latex_issues, strip_option_prefix, repair_correct_answer,
    validate_questions_latex, option_text as _option_text,
    NO_OPTION_TYPES, OPTION_BEARING_TYPES, CORRECT_ANSWER_STRIP_TYPES,
)


def _repair_question(q: dict) -> tuple[dict, list[str]]:
    """Returns (repaired_question, [field names that actually changed])."""
    changed: list[str] = []
    out = dict(q)
    q_type = (q.get("type") or "").lower()

    text = q.get("text") or ""
    new_text = repair_common_latex_issues(text)
    if new_text != text:
        out["text"] = new_text
        changed.append("text")

    explanation = q.get("explanation") or ""
    new_explanation = repair_common_latex_issues(explanation)
    if new_explanation != explanation:
        out["explanation"] = new_explanation
        changed.append("explanation")

    opts = q.get("options")
    # Schema consistency: fill/short/long questions must never carry an
    # options array — same guard added to finalize_generated_questions,
    # applied retroactively to already-stored data here.
    original_opt_texts: list[str] | None = None
    repaired_opt_texts: list[str] | None = None
    if q_type in NO_OPTION_TYPES and opts is not None:
        out["options"] = None
        changed.append("options(nulled: wrong type)")
    elif isinstance(opts, list):
        original_opt_texts = [_option_text(opt) for opt in opts]
        new_opts = []
        repaired_opt_texts = []
        any_opt_changed = False
        for i, opt in enumerate(opts):
            original = _option_text(opt)
            repaired = original
            # Each option's own list position is passed so a leading
            # "A. "/"1. " is only stripped when it's the label a
            # duplicate-prefix bug would actually produce at that position —
            # otherwise genuine content like "A. Einstein" gets corrupted.
            if q_type in OPTION_BEARING_TYPES:
                repaired = strip_option_prefix(repaired, i)
            repaired = repair_common_latex_issues(repaired)
            repaired_opt_texts.append(repaired)
            if repaired != original:
                any_opt_changed = True
            if isinstance(opt, dict):
                new_opts.append({**opt, "text": repaired})
            else:
                new_opts.append(repaired)
        if any_opt_changed:
            out["options"] = new_opts
            changed.append("options")

    for key in ("correct_answer", "correctAnswer"):
        if key in q and q[key] is not None:
            original = str(q[key])
            repaired = original
            if q_type in CORRECT_ANSWER_STRIP_TYPES:
                # Derived from whichever original option it matches (already
                # fully LaTeX-repaired either way — see repair_correct_answer),
                # instead of an independent, unguarded strip.
                repaired = repair_correct_answer(repaired, original_opt_texts, repaired_opt_texts)
            else:
                repaired = repair_common_latex_issues(repaired)
            if repaired != original:
                out[key] = repaired
                changed.append(key)

    pairs = q.get("pairs")
    if isinstance(pairs, list):
        new_pairs = []
        any_pair_changed = False
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                new_pairs.append(pair)
                continue
            new_pair = dict(pair)
            for side in ("left", "right"):
                if side in pair and pair[side] is not None:
                    original = str(pair[side])
                    # Same position-aware label strip as options (see
                    # strip_option_prefix) — a match item can carry the same
                    # duplicate-label artifact an MCQ option does.
                    repaired = repair_common_latex_issues(strip_option_prefix(original, i))
                    if repaired != original:
                        new_pair[side] = repaired
                        any_pair_changed = True
            new_pairs.append(new_pair)
        if any_pair_changed:
            out["pairs"] = new_pairs
            changed.append("pairs")

    return out, changed


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Persist changes (default: dry run)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N assessments")
    args = parser.parse_args()

    async with AsyncSessionLocal() as db:
        stmt = select(PracticeAssessment)
        if args.limit:
            stmt = stmt.limit(args.limit)
        result = await db.execute(stmt)
        assessments = result.scalars().all()

        print(f"[migrate_latex_normalize] {len(assessments)} assessment(s) loaded ({'APPLY' if args.apply else 'DRY RUN'})", flush=True)

        total_questions = 0
        total_repaired_fields = 0
        assessments_touched = 0
        still_failing: list[tuple[str, str, list[str]]] = []  # (assessment_id, question_id, issues)

        for assessment in assessments:
            qjson = assessment.question_json
            if not isinstance(qjson, list) or not qjson:
                continue

            repaired_questions = []
            assessment_changed = False
            diff_lines = []

            for q in qjson:
                if not isinstance(q, dict):
                    repaired_questions.append(q)
                    continue
                total_questions += 1
                repaired_q, changed_fields = _repair_question(q)
                repaired_questions.append(repaired_q)
                if changed_fields:
                    assessment_changed = True
                    total_repaired_fields += len(changed_fields)
                    diff_lines.append(f"    q={q.get('id')} fields={changed_fields}")

            # Re-validate AFTER repair — whatever still fails needs a human
            # or a regeneration, this script deliberately does not guess further.
            issues = await validate_questions_latex(repaired_questions)
            for qid, msgs in issues.items():
                still_failing.append((str(assessment.id), qid, msgs))

            if assessment_changed:
                assessments_touched += 1
                print(f"[migrate_latex_normalize] assessment {assessment.id} ({assessment.title!r}):", flush=True)
                for line in diff_lines:
                    print(line, flush=True)
                if args.apply:
                    assessment.question_json = repaired_questions
                    await db.flush()

        if args.apply:
            await db.commit()

        print(flush=True)
        print("=== Summary ===", flush=True)
        print(f"Assessments scanned:  {len(assessments)}", flush=True)
        print(f"Questions scanned:    {total_questions}", flush=True)
        print(f"Assessments touched:  {assessments_touched}", flush=True)
        print(f"Fields repaired:      {total_repaired_fields}", flush=True)
        print(f"Still failing after repair: {len(still_failing)}", flush=True)
        if still_failing:
            print(flush=True)
            print("Question IDs needing manual review/regeneration:", flush=True)
            for assessment_id, qid, msgs in still_failing:
                print(f"  assessment={assessment_id} question={qid}", flush=True)
                for m in msgs:
                    print(f"      - {m}", flush=True)

        if not args.apply:
            print(flush=True)
            print("Dry run only — no changes written. Re-run with --apply to persist.", flush=True)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
