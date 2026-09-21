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
from app.services.latex_validator import repair_common_latex_issues, validate_questions_latex


def _option_text(opt) -> str:
    if isinstance(opt, dict):
        return str(opt.get("text", ""))
    return str(opt) if opt is not None else ""


def _repair_question(q: dict) -> tuple[dict, list[str]]:
    """Returns (repaired_question, [field names that actually changed])."""
    changed: list[str] = []
    out = dict(q)

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

    for key in ("correct_answer", "correctAnswer"):
        if key in q and q[key] is not None:
            original = str(q[key])
            repaired = repair_common_latex_issues(original)
            if repaired != original:
                out[key] = repaired
                changed.append(key)

    opts = q.get("options")
    if isinstance(opts, list):
        new_opts = []
        any_opt_changed = False
        for opt in opts:
            original = _option_text(opt)
            repaired = repair_common_latex_issues(original)
            if repaired != original:
                any_opt_changed = True
            if isinstance(opt, dict):
                new_opts.append({**opt, "text": repaired})
            else:
                new_opts.append(repaired)
        if any_opt_changed:
            out["options"] = new_opts
            changed.append("options")

    pairs = q.get("pairs")
    if isinstance(pairs, list):
        new_pairs = []
        any_pair_changed = False
        for pair in pairs:
            if not isinstance(pair, dict):
                new_pairs.append(pair)
                continue
            new_pair = dict(pair)
            for side in ("left", "right"):
                if side in pair and pair[side] is not None:
                    original = str(pair[side])
                    repaired = repair_common_latex_issues(original)
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
