"""Regression smoke test: non-AI-Assistant generators, baseline vs new code.

Detects BREAKAGE, not quality: success/failure, output type/shape (JSON keys),
truncation (finish_reason), output length, script/language, emoji fields — and
records exactly what each version SENT to the model (max_tokens, temperature).

Both versions run with identical conditions:
  - generator: OpenAI only (cheapest available; Gemini/Claude hidden), so token
    ceilings and temperatures are observable on the wire;
  - content guard disabled (its classifier is not a feature output);
  - the same fixed inputs.

Run once per code root, then compare:
    .venv/bin/python scripts/regression_smoke.py run --root ../eduverse-backend-baseline --label baseline --out /tmp/base.json
    .venv/bin/python scripts/regression_smoke.py run --root . --label new --out /tmp/new.json
    .venv/bin/python scripts/regression_smoke.py compare /tmp/base.json /tmp/new.json
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini, USD per 1M tokens (verify)

DOC_TEXT = ("Reflection of light: when light falls on a polished surface it bounces back. The angle of "
            "incidence equals the angle of reflection, and the incident ray, reflected ray and normal lie in "
            "the same plane. Plane mirrors form virtual, erect images of the same size.")
SESSION_CTX = {"student_mode": True, "grade": 8, "board": "CBSE", "subject": None, "language": "en",
               "tone": "neutral", "difficulty": "medium", "output_mode": "default"}


def cases(label: str):
    """(name, kind, coroutine factory). kind: json | text | stream."""
    audio_ctx = {"grade": 8, "board": "CBSE"}
    if label == "new":                       # new audio.py merges the request's language into context
        audio_ctx = {**audio_ctx, "language": "hi"}
    return [
        ("assessment_questions", "json", lambda ai: ai.generate_practice_assessment(
            subject="Science", topics=["Reflection of light"], grade=8, board="CBSE", difficulty="medium",
            question_count=3, question_types=["mcq", "short_answer"], mode="practice")),
        ("assignment_questions_json_mode", "json", lambda ai: ai.generate_assignment_questions(
            topic="Reflection of light", subject="Physics", grade=8, mcq_count=2, short_answer_count=1)),
        ("rubric", "json", lambda ai: ai.generate_rubric(board="CBSE", grade=8, subject="Science",
                                                         topic="Reflection of light", criteria_count=3)),
        ("mindmap_en", "json", lambda ai: ai.generate_mindmap(topic="Photosynthesis", subject="Biology", grade=7,
                                                              board="CBSE", depth=3, language="en")),
        ("mindmap_ta", "json", lambda ai: ai.generate_mindmap(topic="Photosynthesis", subject="Biology", grade=7,
                                                              board="CBSE", depth=3, language="ta")),
        ("ebook_outline", "json", lambda ai: ai.generate_ebook_outline(
            title="Light", topic="Reflection of light", subject="Physics", language="en", chapter_range=(3, 4),
            tone="academic", grade=8, board="CBSE")),
        ("career_analysis_en", "json", lambda ai: ai.analyze_career(
            interests=["robotics", "maths"], strengths=["problem solving"], target_careers=["Engineer"], grade=11,
            context={"stream": "Science"}, language="en")),
        ("career_analysis_hi", "json", lambda ai: ai.analyze_career(
            interests=["robotics", "maths"], strengths=["problem solving"], target_careers=["Engineer"], grade=11,
            context={"stream": "Science"}, language="hi")),
        ("audio_qa_hi (router args)", "text", lambda ai: ai.chat(
            messages=[{"role": "user", "content": "What is reflection of light?"}], context=audio_ctx)),
        ("vault_query_ask_document", "text", lambda ai: ai.ask_document(
            query="What does the document say about reflection?", context=DOC_TEXT, ai_context=dict(SESSION_CTX))),
        ("follow_up_questions", "json", lambda ai: ai.generate_follow_up_questions(
            user_message="What is reflection?", ai_response=DOC_TEXT, count=4, language="en")),
        ("playground_stream_experiment_en", "stream", lambda ai: ai.stream_playground(
            topic="Reflection of light", mode="experiment", messages=[], grade=8, harder_mode=False,
            context=dict(SESSION_CTX), language=None)),
        ("playground_stream_challenge_hi", "stream", lambda ai: ai.stream_playground(
            topic="Reflection of light", mode="challenge", messages=[], grade=8, harder_mode=True,
            context={**SESSION_CTX, "language": "hi"}, language="hi")),
        ("playground_setup", "json", lambda ai: ai.playground_setup(topic="Photosynthesis")),
        ("playground_simulate", "json", lambda ai: ai.playground_simulate(
            topic="Photosynthesis", variables={"light_intensity": 70, "co2_level": 40})),
        ("playground_roleplay", "json", lambda ai: ai.playground_roleplay(topic="Photosynthesis", grade=8, board="CBSE")),
        ("playground_imagine", "json", lambda ai: ai.playground_imagine(topic="A world without friction", grade=8)),
        ("playground_timeline", "json", lambda ai: ai.playground_timeline(topic="History of optics", grade=8)),
        ("lesson_plan (Gemini-only path)", "json", lambda ai: ai.generate_lesson_plan(
            class_id="x", topic="Reflection of light", board="CBSE", grade=8, subject="Science")),
    ]


# ── recording proxy for the OpenAI client ────────────────────────────────────

CALLS: list[dict] = []
CURRENT: dict = {"case": None}


class _Completions:
    def __init__(self, real):
        self._real = real

    async def create(self, **kw):
        rec = {"case": CURRENT["case"], "model": kw.get("model"), "max_tokens": kw.get("max_tokens"),
               "temperature": kw.get("temperature"), "json_mode": "response_format" in kw,
               "stream": bool(kw.get("stream"))}
        CALLS.append(rec)
        if kw.get("stream"):
            kw.setdefault("stream_options", {"include_usage": True})
            stream = await self._real.create(**kw)

            async def gen():
                async for ch in stream:
                    if getattr(ch, "usage", None):
                        rec["input_tokens"], rec["output_tokens"] = ch.usage.prompt_tokens, ch.usage.completion_tokens
                    if not ch.choices:          # usage-only chunk: baseline code would IndexError on it
                        continue
                    if ch.choices[0].finish_reason:
                        rec["finish_reason"] = ch.choices[0].finish_reason
                    yield ch
            return gen()
        r = await self._real.create(**kw)
        rec["input_tokens"], rec["output_tokens"] = r.usage.prompt_tokens, r.usage.completion_tokens
        rec["finish_reason"] = r.choices[0].finish_reason
        return r


class _Chat:
    def __init__(self, real):
        self.completions = _Completions(real.completions)


class _Client:
    def __init__(self, real):
        self.chat = _Chat(real.chat)


# ── shape helpers ────────────────────────────────────────────────────────────

def _emoji_like(s: str) -> bool:
    return any(unicodedata.category(c) == "So" or ord(c) >= 0x1F000 for c in s)


def emoji_fields(obj, out=None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and re.search(r"emoji|avatar|icon", k, re.I):
                out.setdefault(k, []).append(v)
            emoji_fields(v, out)
    elif isinstance(obj, list):
        for v in obj:
            emoji_fields(v, out)
    return out


def shape(obj, depth=0):
    if isinstance(obj, dict):
        return {"type": "dict", "keys": sorted(obj.keys())}
    if isinstance(obj, list):
        item_keys = sorted({k for x in obj if isinstance(x, dict) for k in x})
        return {"type": "list", "len": len(obj), "item_keys": item_keys}
    return {"type": type(obj).__name__}


def script_share(text: str) -> dict:
    counts: dict[str, int] = {}
    for ch in text:
        if ch.isalpha():
            try:
                counts[unicodedata.name(ch).split()[0]] = counts.get(unicodedata.name(ch).split()[0], 0) + 1
            except ValueError:
                pass
    total = sum(counts.values()) or 1
    return {k: round(v / total, 2) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:3]}


async def run(root: Path, label: str, out: Path) -> None:
    os.chdir(root)
    sys.path.insert(0, str(root))
    from app.config import settings
    from app.services import ai_service as ai_mod
    from openai import AsyncOpenAI

    assert Path(ai_mod.__file__).resolve().is_relative_to(root.resolve()), "imported the wrong code root"
    settings.CONTENT_GUARD_ENABLED = False
    proxy = _Client(AsyncOpenAI(api_key=settings.OPENAI_API_KEY))
    ai_mod.AIService._get_openai = lambda self: proxy
    for getter in ("_get_gemini_async", "_get_gemini", "_get_anthropic"):
        setattr(ai_mod.AIService, getter, lambda self: None)

    results = []
    for name, kind, factory in cases(label):
        CURRENT["case"] = name
        t0 = time.monotonic()
        rec: dict = {"case": name, "kind": kind}
        try:
            ai = ai_mod.AIService()
            if kind == "stream":
                text = "".join([c async for c in factory(ai)])
                value = text
            else:
                value = await factory(ai)
            rec["ok"] = True
            rec["shape"] = shape(value)
            dumped = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            rec["chars"] = len(dumped)
            rec["scripts"] = script_share(dumped)
            rec["emoji_fields"] = {k: {"values": v[:3], "all_emoji": all(_emoji_like(x) for x in v)}
                                   for k, v in emoji_fields(value).items()}
            rec["head"] = dumped[:300]
        except Exception as exc:
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec["seconds"] = round(time.monotonic() - t0, 1)
        rec["llm_calls"] = [c for c in CALLS if c["case"] == name]
        results.append(rec)
        print(f"[{label}] {name:<36} ok={rec['ok']} calls={len(rec['llm_calls'])} "
              f"finish={[c.get('finish_reason') for c in rec['llm_calls']]} "
              f"max_tokens={[c.get('max_tokens') for c in rec['llm_calls']]}", flush=True)
    tin = sum(c.get("input_tokens") or 0 for c in CALLS)
    tout = sum(c.get("output_tokens") or 0 for c in CALLS)
    summary = {"label": label, "input_tokens": tin, "output_tokens": tout,
               "cost_usd": round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6, 4)}
    print(f"[{label}] tokens in={tin} out={tout} cost≈${summary['cost_usd']}")
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1))


def compare(base_path: Path, new_path: Path) -> None:
    base = {r["case"]: r for r in json.loads(base_path.read_text())["results"]}
    new = {r["case"]: r for r in json.loads(new_path.read_text())["results"]}
    print("| Case | OK base/new | Shape same | Finish base → new | max_tokens base → new | temperature base → new | chars base → new | emoji fields new | Flags |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, b in base.items():
        n = new.get(name, {})
        flags = []
        if b.get("ok") and not n.get("ok"):
            flags.append("NEW FAILURE")
        same_shape = b.get("shape") == n.get("shape")
        if b.get("ok") and n.get("ok") and b.get("shape", {}).get("type") != n.get("shape", {}).get("type"):
            flags.append("TYPE CHANGED")
        if b.get("shape", {}).get("type") == "dict" and n.get("shape", {}).get("type") == "dict":
            missing = set(b["shape"]["keys"]) - set(n["shape"]["keys"])
            if missing:
                flags.append(f"missing keys {sorted(missing)}")
        fin = lambda r: [c.get("finish_reason") for c in r.get("llm_calls", [])]
        mt = lambda r: [c.get("max_tokens") for c in r.get("llm_calls", [])]
        tp = lambda r: [c.get("temperature") for c in r.get("llm_calls", [])]
        if "length" in fin(n):
            flags.append("TRUNCATED (finish=length)")
        nm = [m for m in mt(n) if m]
        if nm and (not [m for m in mt(b) if m] or min(nm) < min([m for m in mt(b) if m] or [10 ** 9])):
            flags.append("LOWER/NEW TOKEN CEILING")
        if tp(b) != tp(n):
            flags.append("temperature changed")
        ef = n.get("emoji_fields") or {}
        bad_emoji = [k for k, v in ef.items() if re.search("emoji|avatar", k, re.I) and not v["all_emoji"]]
        if bad_emoji:
            flags.append(f"non-emoji in {bad_emoji}")
        print(f"| {name} | {b.get('ok')}/{n.get('ok')} | {'yes' if same_shape else 'NO'} | {fin(b)} → {fin(n)} | "
              f"{mt(b)} → {mt(n)} | {tp(b)} → {tp(n)} | {b.get('chars')} → {n.get('chars')} | "
              f"{ {k: v['values'][:2] for k, v in ef.items()} if ef else '–'} | {'; '.join(flags) or 'OK'} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--label", required=True, choices=["baseline", "new"])
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("new")
    a = p.parse_args()
    if a.cmd == "run":
        asyncio.run(run(Path(a.root).resolve(), a.label, Path(a.out).resolve()))
    else:
        compare(Path(a.base), Path(a.new))


if __name__ == "__main__":
    main()
