r"""One-off migration: collapse over-escaped LaTeX backslashes in stored
Assessment Hub questions (practice_assessments.question_json).

An LLM writing LaTeX into a JSON string sometimes escaped the backslash twice,
so "$\theta$" was stored as "$\\theta$". TeX reads "\\" as a LINE BREAK, which
is why affected questions rendered as a stray line break before italic "theta",
"5", a line break and a literal ", kg" in front of \mathrm{kg}, or red KaTeX
error text for "37^\circ".

This applies exactly one transformation — latex_validator.collapse_over_escaped_
commands — to every text field of every question: stem, options, correct answer,
explanation, and both sides of every match pair. It deliberately does NOT run
the broader repair_common_latex_issues (unbalanced $, option-prefix stripping,
stray \%); that is migrate_latex_normalize.py's job, and mixing the two would
make it impossible to tell which change did what if something goes wrong.

SAFETY
  - Dry run by default. Nothing is written without --apply.
  - --apply writes a JSON backup of every row it is about to change BEFORE
    changing it, and refuses to run if the backup cannot be written.
  - --rollback restores from such a backup.
  - Genuine LaTeX line breaks are preserved: every \begin{...}...\end{...}
    environment is compared before and after, and the migration ABORTS if any
    of them changed. Real stored data's 31 multi-backslash runs are all matrix
    row separators, so this check is the one that matters most.
  - Every changed field is re-validated with a real KaTeX parse; anything still
    failing is listed by question id rather than silently accepted.

USAGE (run from the backend directory)
    python -m scripts.migrate_collapse_over_escaped_latex                  # dry run
    python -m scripts.migrate_collapse_over_escaped_latex --samples 25
    python -m scripts.migrate_collapse_over_escaped_latex --apply
    python -m scripts.migrate_collapse_over_escaped_latex --rollback backup.json

TARGETING AN ENVIRONMENT
    Connection comes from --database-url, else $MIGRATION_DATABASE_URL, else
    $DATABASE_URL, else the app's own settings (i.e. whatever .env points at).
    A restored dump in a local database is the intended way to rehearse this
    against production data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import asyncio
import copy
import json
import re
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.assessment import PracticeAssessment
from app.services.latex_validator import (
    collapse_over_escaped_commands, has_over_escaped_command,
    extract_math_segments, option_text, math_environment_spans,
)
from app.services.katex_check import check_segments

# Every text-bearing key a question may carry. Both the snake_case and
# camelCase spellings appear in stored data: the generator writes
# correct_answer/explanation, while the frontend's edit-save path writes
# correctAnswer, so a migration that knew only one spelling would silently
# skip every question that had been edited (or every one that had not).
SCALAR_FIELDS = ("text", "question", "correct_answer", "correctAnswer", "explanation")


# ─── transformation ──────────────────────────────────────────────────────────

# Comparing these before and after a transformation is what proves no genuine
# line break was destroyed; see latex_validator.math_environment_spans.
_env_bodies = math_environment_spans


def collapse_question(q: dict) -> tuple[dict, list[tuple[str, str, str]]]:
    """Returns (new_question, [(field, before, after), ...]) for changed fields."""
    if not isinstance(q, dict):
        return q, []
    out = copy.deepcopy(q)
    changes: list[tuple[str, str, str]] = []

    def apply(value):
        return collapse_over_escaped_commands(value) if isinstance(value, str) else value

    for key in SCALAR_FIELDS:
        before = out.get(key)
        if isinstance(before, str):
            after = apply(before)
            if after != before:
                out[key] = after
                changes.append((key, before, after))

    opts = out.get("options")
    if isinstance(opts, list):
        new_opts = []
        for i, opt in enumerate(opts):
            if isinstance(opt, str):
                after = apply(opt)
                if after != opt:
                    changes.append((f"options[{i}]", opt, after))
                new_opts.append(after)
            elif isinstance(opt, dict) and isinstance(opt.get("text"), str):
                new_opt = dict(opt)
                after = apply(opt["text"])
                if after != opt["text"]:
                    changes.append((f"options[{i}].text", opt["text"], after))
                new_opt["text"] = after
                new_opts.append(new_opt)
            else:
                new_opts.append(opt)
        out["options"] = new_opts

    pairs = out.get("pairs")
    if isinstance(pairs, list):
        new_pairs = []
        for i, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                new_pairs.append(pair)
                continue
            new_pair = dict(pair)
            for side in ("left", "right"):
                before = pair.get(side)
                if isinstance(before, str):
                    after = apply(before)
                    if after != before:
                        changes.append((f"pairs[{i}].{side}", before, after))
                    new_pair[side] = after
            new_pairs.append(new_pair)
        out["pairs"] = new_pairs

    return out, changes


def all_text(q: dict) -> list[str]:
    """Every text value of a question, for environment/KaTeX comparison."""
    if not isinstance(q, dict):
        return []
    values = [q.get(k) for k in SCALAR_FIELDS]
    for opt in q.get("options") or []:
        values.append(option_text(opt) if not isinstance(opt, str) else opt)
    for pair in q.get("pairs") or []:
        if isinstance(pair, dict):
            values += [pair.get("left"), pair.get("right")]
    return [v for v in values if isinstance(v, str)]


# ─── driver ──────────────────────────────────────────────────────────────────

def resolve_database_url(explicit: str | None) -> tuple[str, str]:
    """(url, where_it_came_from). Never silently guesses."""
    if explicit:
        return explicit, "--database-url"
    for var in ("MIGRATION_DATABASE_URL", "DATABASE_URL"):
        value = os.environ.get(var)
        if value:
            return value, f"${var}"
    from app.config import settings
    for attr in ("DATABASE_URL", "SQLALCHEMY_DATABASE_URI", "ASYNC_DATABASE_URL"):
        value = getattr(settings, attr, None)
        if value:
            return str(value), f"app settings ({attr})"
    return (
        f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}",
        "app settings (DB_* parts)",
    )


def redact(url: str) -> str:
    return re.sub(r"://([^:/@]+):[^@]*@", r"://\1:***@", url)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--rollback", metavar="BACKUP_FILE", help="restore from a backup and exit")
    ap.add_argument("--database-url", help="target database (default: env, then app settings)")
    ap.add_argument("--backup-file", help="where to write the pre-change backup")
    ap.add_argument("--limit", type=int, help="only examine the first N assessments")
    ap.add_argument("--samples", type=int, default=10, help="before/after samples to print")
    args = ap.parse_args()

    url, source = resolve_database_url(args.database_url)
    engine = create_async_engine(url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    print("=" * 78)
    print("Collapse over-escaped LaTeX backslashes — practice_assessments.question_json")
    print("=" * 78)
    print(f"database : {redact(url)}")
    print(f"           (from {source})")
    print(f"mode     : {'ROLLBACK' if args.rollback else ('APPLY (writes)' if args.apply else 'DRY RUN (no writes)')}")
    print()

    try:
        if args.rollback:
            return await do_rollback(Session, args.rollback)
        return await do_migration(Session, args, url)
    finally:
        await engine.dispose()


async def do_rollback(Session, path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        backup = json.load(fh)
    rows = backup.get("rows", [])
    print(f"Restoring {len(rows)} assessment(s) from {path}")
    print(f"backup taken: {backup.get('taken_at')}  source: {redact(backup.get('database', ''))}")
    async with Session() as db:
        for row in rows:
            await db.execute(
                update(PracticeAssessment)
                .where(PracticeAssessment.id == row["id"])
                .values(question_json=row["question_json"])
            )
        await db.commit()
    print(f"\nRestored {len(rows)} assessment(s).")
    return 0


async def do_migration(Session, args, url: str) -> int:
    async with Session() as db:
        stmt = select(PracticeAssessment).order_by(PracticeAssessment.created_at.desc())
        if args.limit:
            stmt = stmt.limit(args.limit)
        assessments = (await db.execute(stmt)).scalars().all()

        rows_scanned = len(assessments)
        questions_scanned = 0
        questions_changed = 0
        fields_changed = 0
        envs_before_total = 0
        envs_after_total = 0
        env_violations: list[str] = []
        samples: list[tuple[str, str, str, str, str, str]] = []
        changed_rows: list[tuple] = []          # (assessment, new_question_json)
        backup_rows: list[dict] = []
        post_segments: dict[str, list[tuple[str, str, str, bool]]] = {}
        residual: list[str] = []

        for a in assessments:
            qjson = a.question_json
            if isinstance(qjson, str):
                try:
                    qjson = json.loads(qjson)
                except json.JSONDecodeError:
                    continue
            if not isinstance(qjson, list):
                continue

            new_qjson = []
            row_changed = False
            for q in qjson:
                questions_scanned += 1
                new_q, changes = collapse_question(q)

                # Environment (line-break) preservation check.
                before_envs = [e for t in all_text(q) for e in _env_bodies(t)]
                after_envs = [e for t in all_text(new_q) for e in _env_bodies(t)]
                envs_before_total += len(before_envs)
                envs_after_total += len(after_envs)
                if sorted(before_envs) != sorted(after_envs):
                    env_violations.append(
                        f"assessment {a.id} question {q.get('id')!r}: "
                        f"{len(before_envs)} env(s) before, {len(after_envs)} after"
                    )

                if changes:
                    row_changed = True
                    questions_changed += 1
                    fields_changed += len(changes)
                    for field, before, after in changes:
                        if len(samples) < args.samples:
                            samples.append(
                                (str(a.id), a.title or "", str(q.get("id")), field, before, after)
                            )
                    if has_over_escaped_command(" ".join(all_text(new_q))):
                        residual.append(f"assessment {a.id} question {q.get('id')!r}")

                # KaTeX-validate EVERY question's post-transform state, not just
                # the changed ones. A question this migration did not touch can
                # still be broken for an unrelated reason, and the point of the
                # list is "what needs a human", not "what did I break".
                for t in all_text(new_q):
                    for seg in extract_math_segments(t):
                        post_segments.setdefault(seg, []).append(
                            (str(a.id), a.title or "", str(q.get("id")), bool(changes))
                        )
                new_qjson.append(new_q)

            if row_changed:
                changed_rows.append((a, new_qjson))
                backup_rows.append({"id": str(a.id), "title": a.title,
                                    "question_json": qjson})

        # Real KaTeX parse of every post-migration math segment.
        katex_errors = await check_segments(list(post_segments))
        # (assessment_id, title, question_id, was_changed) -> [error, ...]
        failing: dict[tuple, list[str]] = {}
        if katex_errors:
            for seg, msg in katex_errors.items():
                for owner in post_segments.get(seg, []):
                    failing.setdefault(owner, []).append(f"{msg[:130]} | segment {seg!r}")
        failing_assessments = {k[0] for k in failing}
        failing_changed = {k for k in failing if k[3]}

        # ── report ──────────────────────────────────────────────────────────
        print("─" * 78)
        print("DRY-RUN REPORT" if not args.apply else "APPLY REPORT")
        print("─" * 78)
        print(f"  assessments scanned            : {rows_scanned}")
        print(f"  questions scanned              : {questions_scanned}")
        print(f"  assessments to change          : {len(changed_rows)}")
        print(f"  questions changed              : {questions_changed}")
        print(f"  fields changed                 : {fields_changed}")
        print(f"  \\begin{{}} environments before  : {envs_before_total}")
        print(f"  \\begin{{}} environments after   : {envs_after_total}")
        print(f"  environments ALTERED           : {len(env_violations)}  (must be 0)")
        if katex_errors is None:
            print(f"  questions failing KaTeX        : (checker unavailable — install node+katex)")
        else:
            print(f"  questions failing KaTeX        : {len(failing)}"
                  f"  across {len(failing_assessments)} assessment(s)")
            print(f"     ...of which this migration changed: {len(failing_changed)}")
        print(f"  over-escapes still present     : {len(residual)}  (must be 0)")
        print()

        if env_violations:
            print("!! ENVIRONMENT VIOLATIONS — a LaTeX line break would be destroyed:")
            for v in env_violations[:20]:
                print(f"   {v}")
            print("\nABORTING. No changes written.")
            return 2

        if failing:
            print("─" * 78)
            print("NEEDS MANUAL REVIEW OR REGENERATION")
            print("These questions still fail a REAL KaTeX parse after the collapse.")
            print("This migration cannot fix them — it only un-doubles backslashes.")
            print("─" * 78)
            for (aid, title, qid, was_changed) in sorted(failing, key=lambda k: (k[1], k[2])):
                tag = "changed by this migration" if was_changed else "pre-existing, untouched"
                print(f"  question id : {qid}")
                print(f"  assessment  : {aid}  [{title[:50]}]")
                print(f"  status      : {tag}")
                for m in failing[(aid, title, qid, was_changed)][:3]:
                    print(f"  error       : {m}")
                print()
            print(f"  TOTAL: {len(failing)} question(s) across "
                  f"{len(failing_assessments)} assessment(s) need manual attention.")
            print(f"  Question IDs: {', '.join(sorted(k[2] for k in failing))}")
            print()

        if residual:
            print("Questions STILL containing an over-escaped backslash:")
            for r in residual[:25]:
                print(f"   {r}")
            print()

        if samples:
            print(f"── {len(samples)} before/after sample(s) ─────────────────────────────────")
            for aid, title, qid, field, before, after in samples:
                print(f"\n  [{title[:40]}] question {qid} field {field}")
                print(f"     before: {before[:180]!r}")
                print(f"     after : {after[:180]!r}")
            print()
        else:
            print("No over-escaped LaTeX found — nothing to change.\n")

        if not changed_rows:
            return 0

        if not args.apply:
            print("DRY RUN — nothing written. Re-run with --apply to persist,")
            print("and keep the backup file it writes.")
            return 0

        # ── apply ────────────────────────────────────────────────────────────
        backup_path = args.backup_file or (
            f"backup_over_escaped_latex_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        )
        payload = {
            "taken_at": datetime.now(timezone.utc).isoformat(),
            "database": redact(url),
            "script": os.path.basename(__file__),
            "rows": backup_rows,
        }
        try:
            with open(backup_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1)
        except OSError as e:
            print(f"!! Could not write backup to {backup_path}: {e}")
            print("ABORTING — refusing to change rows without a backup.")
            return 3
        print(f"Backup of {len(backup_rows)} row(s) written to: {backup_path}")
        print(f"Roll back with:\n"
              f"  python -m scripts.migrate_collapse_over_escaped_latex --rollback {backup_path}\n")

        for a, new_qjson in changed_rows:
            await db.execute(
                update(PracticeAssessment)
                .where(PracticeAssessment.id == a.id)
                .values(question_json=new_qjson)
            )
        await db.commit()
        print(f"Applied to {len(changed_rows)} assessment(s), {questions_changed} question(s).")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
