r"""Read-only health check for assessment LaTeX. Writes nothing, ever.

Answers, for whichever database you point it at:
  1. Is real KaTeX validation working here? (If not, newly generated questions
     are only being checked structurally.)
  2. How many stored questions are broken, and in which of three distinct ways —
     because each needs a different remedy:

       over-escaped   "$\\theta$"                 -> the collapse migration fixes it
       corrupted      "$^{10}<PROTECT marker>..." -> text was LOST; --repair-corrupted
                                                     or regenerate
       katex-failing  "$4^2 = ___$"               -> genuinely malformed; needs a human

Connection resolution matches the migration script: --database-url, then
$MIGRATION_DATABASE_URL, then $DATABASE_URL, then app settings.

    python -m scripts.check_latex_health
    python -m scripts.check_latex_health --verbose
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import asyncio
import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.models.assessment import PracticeAssessment
from app.services.katex_check import check_segments, probe
from app.services.latex_validator import (
    contains_sentinel, describe_control_characters, extract_math_segments,
    has_control_characters, has_over_escaped_command,
)
from scripts.migrate_collapse_over_escaped_latex import (
    all_text, redact, resolve_database_url,
)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database-url")
    ap.add_argument("--verbose", action="store_true", help="list every affected question")
    args = ap.parse_args()

    url, source = resolve_database_url(args.database_url)
    print("=" * 74)
    print("Assessment LaTeX health check (read-only)")
    print("=" * 74)
    print(f"database : {redact(url)}  (from {source})")

    status = await probe()
    active = status["katex_validation"] == "active"
    print(f"\n1. Real KaTeX validation: {status['katex_validation'].upper()}")
    print(f"   {status['katex_validation_detail']}")
    if not active:
        print("   WARNING: newly generated questions are being checked with structural")
        print("   rules only. Control characters, protection markers, unbalanced braces")
        print("   and unbalanced $ are still caught; malformed math that only a real")
        print("   parser can detect is NOT.")

    engine = create_async_engine(url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with Session() as db:
            rows = (await db.execute(select(PracticeAssessment))).scalars().all()

            total_q = 0
            over, corrupt = [], []
            seg_owners: dict[str, list[tuple]] = {}
            for a in rows:
                qj = a.question_json
                if isinstance(qj, str):
                    try:
                        qj = json.loads(qj)
                    except json.JSONDecodeError:
                        continue
                if not isinstance(qj, list):
                    continue
                for q in qj:
                    if not isinstance(q, dict):
                        continue
                    total_q += 1
                    texts = all_text(q)
                    who = (str(a.id), (a.title or "")[:40], str(q.get("id")))
                    if any(has_over_escaped_command(t) for t in texts):
                        over.append(who)
                    bad = [t for t in texts
                           if has_control_characters(t) or contains_sentinel(t)]
                    if bad:
                        corrupt.append(who + (describe_control_characters(bad[0])
                                              or "protection marker",))
                    for t in texts:
                        for seg in extract_math_segments(t):
                            seg_owners.setdefault(seg, []).append(who)

            katex_bad: dict = {}
            errors = await check_segments(list(seg_owners))
            if errors:
                for seg, msg in errors.items():
                    for who in seg_owners.get(seg, []):
                        katex_bad.setdefault(who, []).append(f"{msg[:110]} | {seg!r}")

            print(f"\n2. Stored content: {len(rows)} assessment(s), {total_q} question(s)")
            print(f"   over-escaped backslashes : {len(over):4d}  -> migrate_collapse_over_escaped_latex.py")
            print(f"   corrupted (markers/ctrl) : {len(corrupt):4d}  -> ...--repair-corrupted, or regenerate")
            if errors is None:
                print(f"   failing a KaTeX parse    :  n/a  (checker unavailable)")
            else:
                print(f"   failing a KaTeX parse    : {len(katex_bad):4d}  -> needs a human")

            healthy = not over and not corrupt and not katex_bad
            print(f"\n   {'ALL CLEAR' if healthy else 'ACTION NEEDED'}")

            if args.verbose:
                for label, items in (("OVER-ESCAPED", over), ("CORRUPTED", corrupt)):
                    if items:
                        print(f"\n   {label}:")
                        for it in items[:60]:
                            print(f"     q={it[2]:8s} {it[0]} [{it[1]}]"
                                  + (f"  {it[3]}" if len(it) > 3 else ""))
                if katex_bad:
                    print("\n   KATEX-FAILING:")
                    for who, msgs in list(katex_bad.items())[:60]:
                        print(f"     q={who[2]:8s} {who[0]} [{who[1]}]")
                        print(f"        {msgs[0]}")
            elif over or corrupt or katex_bad:
                print("\n   Re-run with --verbose to list the affected questions.")
        return 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
