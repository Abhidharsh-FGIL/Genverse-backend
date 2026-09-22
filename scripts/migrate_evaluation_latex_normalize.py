"""
One-off migration: repair and validate LaTeX in existing EvaluationQuestion
rows — the source data behind the Evaluation Hub's Question Bank, paper
detail view, the emailed /take-assessment student exam flow, and the
per-student assessment report.

Sibling script to scripts/migrate_latex_normalize.py (personal Assessment
Hub) — this is the Evaluation Hub equivalent, added because that system's
generation path (AIService.generate_evaluation_paper / evaluation.py
save_paper) had NO repair/validation safety net at all until this same
session's fix, unlike the personal hub which already had one.

Applies the same conservative repairs: unbalanced leading/trailing $ on a
short option, double-escaped \\ce macros, a stray \\% outside math, and a
stray "A. "/"B. " option-label prefix. Then re-validates every question
(balanced $/$$, balanced braces, bare chemical formulas needing \\ce{...})
and reports which question IDs still fail.

Also flags (does not touch) the two-shape "options" inconsistency the
research audit found (array from AI generation vs. {A,B,C,D} dict from
manual entry) — out of scope for a LaTeX migration, listed for awareness.

DRY RUN BY DEFAULT — prints a diff summary and the still-failing list,
writes nothing. Pass --apply to persist.

Run from backend directory:
    python -m scripts.migrate_evaluation_latex_normalize            # dry run
    python -m scripts.migrate_evaluation_latex_normalize --apply     # write changes
    python -m scripts.migrate_evaluation_latex_normalize --limit 50  # sample first N rows
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import asyncio

from sqlalchemy import select

from app.database import AsyncSessionLocal, engine
from app.models.evaluation import EvaluationQuestion
from app.services.latex_validator import (
    repair_common_latex_issues, strip_option_prefix, repair_correct_answer,
    validate_questions_latex, NO_OPTION_TYPES, OPTION_BEARING_TYPES,
    CORRECT_ANSWER_STRIP_TYPES,
)


def _label_index(key: str) -> int | None:
    """Parses a {A:.., B:..} dict's KEY into its 0-based label position
    ("A"/"a" -> 0, "B"/"b" -> 1, ..., "1" -> 0, "2" -> 1, ...). None if the
    key isn't a recognizable single letter or number (e.g. a custom label),
    in which case the caller falls back to raw iteration order."""
    key = key.strip()
    if len(key) == 1 and key.isalpha():
        return ord(key.upper()) - ord("A")
    if key.isdigit():
        return int(key) - 1
    return None


def _option_list_and_shape(options):
    """EvaluationQuestion.options can be a plain array, a {A:.., B:..} dict,
    or (for match questions) {"options": [...], "pairs": [...]}. Returns
    (list_of_option_texts, shape, label_indices) where shape lets the caller
    rebuild the same structure after repairing the text, and label_indices
    (only for "letter_dict") gives each option's position AS IMPLIED BY ITS
    OWN KEY — not raw `dict.values()` iteration order, which happens to
    match Python's guaranteed key-insertion order but has no guaranteed
    correspondence to A/B/C/D at all (a dict rebuilt or edited out of order
    would silently misalign position-based label-stripping otherwise)."""
    if options is None:
        return None, None, None
    if isinstance(options, list):
        return [str(o) for o in options], "list", None
    if isinstance(options, dict):
        if "options" in options and isinstance(options["options"], list):
            return [str(o) for o in options["options"]], "match_dict", None
        # {A: "...", B: "...", ...} shape
        keys = list(options.keys())
        return [str(v) for v in options.values()], "letter_dict", [_label_index(k) for k in keys]
    return None, None, None


def _rebuild_options(original, shape, repaired_texts: list[str]):
    if shape == "list":
        return repaired_texts
    if shape == "match_dict":
        return {**original, "options": repaired_texts}
    if shape == "letter_dict":
        keys = list(original.keys())
        return dict(zip(keys, repaired_texts))
    return original


def _repair_question(q: EvaluationQuestion) -> list[str]:
    """Mutates `q` in place (caller decides whether to persist) and returns
    the list of field names that actually changed."""
    changed: list[str] = []
    q_type = (q.question_type or "").lower()

    text = q.question_text or ""
    new_text = repair_common_latex_issues(text)
    if new_text != text:
        q.question_text = new_text
        changed.append("question_text")

    explanation = q.explanation or ""
    new_explanation = repair_common_latex_issues(explanation)
    if new_explanation != explanation:
        q.explanation = new_explanation or None
        changed.append("explanation")

    original_opt_texts: list[str] | None = None
    repaired_opt_texts: list[str] | None = None

    if q_type in NO_OPTION_TYPES and q.options is not None:
        q.options = None
        changed.append("options(nulled: wrong type)")
    elif q_type in OPTION_BEARING_TYPES and q.options is not None:
        opt_texts, shape, label_indices = _option_list_and_shape(q.options)
        if opt_texts is not None:
            original_opt_texts = opt_texts
            repaired_opt_texts = []
            any_changed = False
            for i, original in enumerate(opt_texts):
                # Position-aware: a leading "A. "/"1. " is only stripped
                # when it's the label a duplicate-prefix bug would actually
                # produce at this position — otherwise genuine content like
                # "A. Einstein" gets corrupted (see strip_option_prefix).
                # For a {A:..,B:..} dict, use the option's OWN key-derived
                # position rather than raw iteration order (see
                # _option_list_and_shape).
                position = label_indices[i] if label_indices and label_indices[i] is not None else i
                repaired = strip_option_prefix(original, position)
                repaired = repair_common_latex_issues(repaired)
                if repaired != original:
                    any_changed = True
                repaired_opt_texts.append(repaired)
            if any_changed:
                q.options = _rebuild_options(q.options, shape, repaired_opt_texts)
                changed.append("options")

    # Match questions store their pairs INSIDE the same options JSONB column
    # (save_paper writes {"options": [...], "pairs": [...]}) rather than as
    # a separate column like the personal Assessment Hub's question_json —
    # repair each pair's left/right text the same way its sibling script
    # (migrate_latex_normalize.py) repairs the personal hub's top-level
    # "pairs" field, so a broken $ delimiter in a match pair doesn't ship
    # unrepaired to students taking the exam.
    if q_type == "match" and isinstance(q.options, dict) and isinstance(q.options.get("pairs"), list):
        new_pairs = []
        any_pair_changed = False
        for i, pair in enumerate(q.options["pairs"]):
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
            q.options = {**q.options, "pairs": new_pairs}
            changed.append("options(pairs)")

    if q.correct_answer is not None:
        original = str(q.correct_answer)
        repaired = original
        if q_type in CORRECT_ANSWER_STRIP_TYPES:
            # Derived from whichever original option it matches (already
            # fully LaTeX-repaired either way — see repair_correct_answer),
            # instead of an independent, unguarded strip.
            repaired = repair_correct_answer(repaired, original_opt_texts, repaired_opt_texts)
        else:
            repaired = repair_common_latex_issues(repaired)
        if repaired != original:
            q.correct_answer = repaired
            changed.append("correct_answer")

    return changed


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Persist changes (default: dry run)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N questions")
    args = parser.parse_args()

    async with AsyncSessionLocal() as db:
        stmt = select(EvaluationQuestion)
        if args.limit:
            stmt = stmt.limit(args.limit)
        result = await db.execute(stmt)
        questions = result.scalars().all()

        print(f"[migrate_evaluation_latex_normalize] {len(questions)} question(s) loaded ({'APPLY' if args.apply else 'DRY RUN'})", flush=True)

        total_repaired_fields = 0
        questions_touched = 0
        options_shape_mismatch: list[str] = []
        still_failing: list[tuple[str, list[str]]] = []

        validate_batch = []
        for q in questions:
            changed_fields = _repair_question(q)
            if changed_fields:
                questions_touched += 1
                total_repaired_fields += len(changed_fields)
                print(f"[migrate_evaluation_latex_normalize] question {q.id} (paper={q.paper_id}): fields={changed_fields}", flush=True)

            opt_texts, shape, _label_indices = _option_list_and_shape(q.options)
            if shape == "letter_dict" and (q.question_type or "").lower() == "mcq":
                options_shape_mismatch.append(str(q.id))

            validate_batch.append({
                "id": str(q.id),
                "text": q.question_text,
                "explanation": q.explanation,
                "correct_answer": q.correct_answer,
                "options": opt_texts,
            })

        issues = await validate_questions_latex(validate_batch)
        for qid, msgs in issues.items():
            still_failing.append((qid, msgs))

        if args.apply:
            await db.commit()

        print(flush=True)
        print("=== Summary ===", flush=True)
        print(f"Questions scanned:    {len(questions)}", flush=True)
        print(f"Questions touched:    {questions_touched}", flush=True)
        print(f"Fields repaired:      {total_repaired_fields}", flush=True)
        print(f"Still failing after repair: {len(still_failing)}", flush=True)
        if still_failing:
            print(flush=True)
            print("Question IDs needing manual review/regeneration:", flush=True)
            for qid, msgs in still_failing:
                print(f"  question={qid}", flush=True)
                for m in msgs:
                    print(f"      - {m}", flush=True)

        if options_shape_mismatch:
            print(flush=True)
            print(
                f"NOTE (out of scope for this LaTeX migration): {len(options_shape_mismatch)} MCQ question(s) "
                "store options as a {A:..,B:..} dict rather than an array — this shape breaks "
                "answer-shuffling (_shuffle_questions_and_options requires the dict shape and silently "
                "no-ops otherwise); flagging for awareness only, not modified here:", flush=True,
            )
            for qid in options_shape_mismatch[:20]:
                print(f"  question={qid}", flush=True)
            if len(options_shape_mismatch) > 20:
                print(f"  ... and {len(options_shape_mismatch) - 20} more", flush=True)

        if not args.apply:
            print(flush=True)
            print("Dry run only — no changes written. Re-run with --apply to persist.", flush=True)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
