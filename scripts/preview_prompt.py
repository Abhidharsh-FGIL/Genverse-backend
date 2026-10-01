"""Dev-only: print the exact AI Assistant prompt for any parameter combination.

No LLM is called. Uses the same normalisers and builder as the live chat path
(app.schemas.ai_payloads + app.prompts.builder), so what you see is what the
provider receives.

    .venv/bin/python scripts/preview_prompt.py --grade 8 --board ICSE --lang ta \
        --student-mode --length detailed --difficulty hard --personality coach \
        --subject Science [--examples] [--three-ways] [--provider gemini|claude|openai] \
        [--question "..."] [--messages]

Plan PLAN_AI_TAILORING.md §8.1.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re  # noqa: E402

from app.prompts.builder import build_system_prompt, split_sections, to_provider_messages  # noqa: E402
from app.prompts.builder import effective_student_mode  # noqa: E402
from app.prompts.params import generation_params  # noqa: E402
from app.config import settings  # noqa: E402
from app.schemas.ai_payloads import normalize_context, normalize_settings  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grade", type=int)
    p.add_argument("--board")
    p.add_argument("--subject")
    p.add_argument("--lang", default="en")
    p.add_argument("--student-mode", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--length", default="summary")
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--personality", default="mentor")
    p.add_argument("--examples", action="store_true")
    p.add_argument("--three-ways", action="store_true")
    p.add_argument("--provider", choices=["gemini", "claude", "openai"], default="gemini")
    p.add_argument("--question", default="What is photosynthesis?")
    p.add_argument("--files", action="store_true", help="treat as a document (RAG) turn — affects params")
    p.add_argument("--messages", action="store_true", help="also print the provider message structure")
    a = p.parse_args()

    ctx = normalize_context({
        "student_mode": a.student_mode, "grade": a.grade, "board": a.board,
        "subject": a.subject, "language": a.lang,
    })
    cs = normalize_settings({
        "student_mode": a.student_mode, "difficulty": a.difficulty,
        "content_length": a.length, "personality": a.personality,
        "explain_3ways": a.three_ways, "examples": a.examples,
    })
    system_prompt = build_system_prompt(ctx, cs, a.provider, question=a.question)

    print("=" * 30, "PARAMETERS (normalised)", "=" * 30)
    print(f"context : {ctx.model_dump(include={'student_mode', 'grade', 'board', 'subject', 'language'})}"
          f"  board_key={ctx.board_key}")
    print(f"settings: {cs.model_dump(include={'student_mode', 'difficulty', 'content_length', 'personality', 'explain_3ways', 'examples'})}")
    print("=" * 30, f"SYSTEM PROMPT ({len(system_prompt)} chars, provider={a.provider})", "=" * 30)
    print(system_prompt)

    if a.messages:
        turns = to_provider_messages([{"role": "user", "content": a.question}], ctx.language or "en")
        print("=" * 30, f"PROVIDER TURNS ({a.provider})", "=" * 30)
        if a.provider == "gemini":
            print("[gemini: config.system_instruction=<system prompt above>; contents=]")
        elif a.provider == "claude":
            print("[claude: system=<system prompt above>; messages=]")
        else:
            print("[openai: messages=[{system: <system prompt above>}] +]")
        for t in turns:
            role = "model" if (a.provider == "gemini" and t["role"] == "assistant") else t["role"]
            print(f"  {role}: {t['content']}")
    sections = split_sections(system_prompt)
    print("=" * 30, "CHECKS", "=" * 30)
    checks = {
        "exactly one Length line": system_prompt.count("Length:") == 1,
        'no "TURNED OFF"/"is OFF" text': not re.search(r"turned off|is off\b|currently off", system_prompt, re.I),
        "no emoji permission (only the ban)": system_prompt.lower().count("emoji") == 1 and "Do not use emojis." in system_prompt,
        "learner_profile only in student mode": ("learner_profile" in sections) == bool(a.student_mode and (a.grade or a.board or a.subject)),
        "precedence present once": system_prompt.count("<precedence>") == 1,
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  sections: {', '.join(sections)}")
    params = generation_params(ctx, cs, a.question, a.files, effective_student_mode(ctx, cs))
    print("=" * 30, "GEN PARAMS", "=" * 30)
    print(f"  max_tokens={params.max_tokens} (safety ceiling)  temperature={params.temperature} "
          f"→ sent: gemini {settings.AI_PRIMARY_MODEL}={params.temperature_for('gemini', settings.AI_PRIMARY_MODEL)}, "
          f"openai {settings.AI_FALLBACK_MODEL}={params.temperature_for('openai', settings.AI_FALLBACK_MODEL)}, "
          f"claude={params.temperature_for('claude', settings.AI_DOCUMENT_MODEL)}")


if __name__ == "__main__":
    main()
