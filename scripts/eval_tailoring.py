"""Live tailoring eval for the AI Assistant (plan §8.3 + amendments 4, 5, 7).

Sends the same questions across parameter combinations through the REAL
generation path (AIService.stream_chat, default provider routing, content
guard on), scores every answer, and writes eval_results/<run>/:
    raw.jsonl        one record per generation (prompt params, answer, tokens, scores)
    outputs/*.md     each answer, for reading
    REPORT.md        per-dimension averages, per-case table, lowest cases,
                     probes, tokens/finish_reason, cost, language calibration

Router-faithful details:
  - canned greetings: the router answers exact greetings like "hi" without an
    LLM when the language is English; the eval does the same (marked "canned").
  - RAG probes: sources are injected with the router's own helpers in the
    router's order (vault excerpts → attached file), so audit G2 behaviour is
    reproduced exactly. Vault retrieval is simulated with fixed excerpts (no
    FAISS/DB needed).

Scoring:
  - automatic: length vs app.prompts.matrices targets (±25%), script purity,
    LaTeX validity (real KaTeX via app.services.katex_check), greeting sentences,
    RAG source markers.
  - LLM judge (Claude, --judge-model, default claude-sonnet-5-5; default sampling — `temperature` is deprecated for it): grade appropriateness,
    board alignment, difficulty fit, language/script purity, LaTeX validity,
    personality fit — 1-5, or null when not applicable. The judge gets the
    syllabus scope table (plan §3.4).

Usage:
    .venv/bin/python scripts/eval_tailoring.py --combos 1,4,7,12,13,18 --probes all \
        --out eval_results/baseline [--concurrency 3]
    .venv/bin/python scripts/eval_tailoring.py --report-only eval_results/baseline
"""
import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.prompts.matrices import (  # noqa: E402
    GREETING_MAX_SENTENCES, LENGTH_TOLERANCE, grade_band, target_words,
)
from app.routers.ai_assistant import (  # noqa: E402
    _get_canned_response, _inject_inline_docs, _inject_rag,
)
from app.services import ai_service as ai_mod  # noqa: E402
from app.services import katex_check  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Cases (plan §8.3 table; numbering kept)
# ─────────────────────────────────────────────────────────────────────────────

QUESTIONS = {
    "photosynthesis": "What is photosynthesis?",
    "quadratic": "Solve x² − 5x + 6 = 0",
    "newton2": "Explain Newton's second law",
    "greeting": "hi",
}


def _c(sm, grade, board, lang, length, difficulty, personality, subject=None, examples=False, three_ways=False):
    return {
        "context": {"student_mode": sm, "grade": grade, "board": board, "subject": subject, "language": lang},
        "settings": {"student_mode": sm, "content_length": length, "difficulty": difficulty,
                     "personality": personality, "examples": examples, "explain_3ways": three_ways},
    }


COMBOS: dict[str, dict] = {
    "1": _c(True, 2, "State Board", "en", "small", "easy", "mentor"),
    "2": _c(True, 2, "CBSE", "ta", "summary", "medium", "friend"),
    "3": _c(True, 3, "IB", "en", "detailed", "expert", "coach"),
    "4": _c(True, 5, "CBSE", "hi", "summary", "medium", "mentor", three_ways=True),
    "5": _c(True, 5, "ICSE", "en", "detailed", "hard", "professor"),
    "6": _c(True, 5, "Cambridge", "ar", "small", "easy", "friend"),
    "7": _c(True, 8, "ICSE", "ta", "detailed", "hard", "coach", subject="Science", examples=True),
    "8": _c(True, 8, "State Board", "en", "summary", "medium", "mentor"),
    "9": _c(True, 8, "IB", "hi", "deep-dive", "medium", "technical-expert"),
    "10": _c(True, 8, "CBSE", "en", "small", "expert", "professor"),
    "11": _c(True, 9, "Cambridge", "en", "detailed", "medium", "professor", subject="Physics"),
    "12": _c(True, 10, "CBSE", "en", "detailed", "hard", "mentor"),
    "13": _c(True, 11, "CBSE", "en", "deep-dive", "expert", "professor", subject="Mathematics"),
    "14": _c(True, 11, "IB", "ar", "detailed", "medium", "mentor"),
    "15": _c(True, 11, "ICSE", "hi", "summary", "hard", "coach"),
    "16": _c(True, 12, "Cambridge", "ta", "deep-dive", "expert", "technical-expert", three_ways=True),
    "17": _c(True, 12, "State Board", "en", "summary", "easy", "friend", examples=True),
    # Student mode OFF: grade/board are what the frontend still sends today (G9).
    "18": _c(False, 8, "CBSE", "en", "summary", "hard", "technical-expert"),
    "19": _c(False, None, None, "hi", "detailed", "easy", "friend"),
    "20": _c(False, None, None, "ar", "deep-dive", "expert", "professor"),
}

# RAG probe sources — written ABOVE the learners' level on purpose, each with a
# distinctive marker phrase so we can detect which sources reached the answer.
VAULT_EXCERPT = (
    "Notes — Light-dependent reactions. Photosystem II (P680) absorbs photons and "
    "drives the photolysis of water, releasing O2; electrons pass through the "
    "plastoquinone pool and cytochrome b6f complex to photosystem I (P700). The "
    "proton-motive force drives ATP synthase (photophosphorylation), and "
    "ferredoxin-NADP+ reductase produces NADPH. Our lab, experiment A: spinach leaf discs "
    "in bicarbonate solution floated after 7 minutes under the LED lamp."
)
ATTACHED_DOC = {
    "filename": "calvin_cycle_notes.pdf",
    "text": (
        "The Calvin–Benson cycle (light-independent reactions) fixes CO2 in the stroma. "
        "The enzyme Rubisco carboxylates ribulose-1,5-bisphosphate (RuBP), producing two "
        "molecules of 3-phosphoglycerate. Using ATP and NADPH from the light reactions, "
        "3-PGA is reduced to glyceraldehyde-3-phosphate (G3P); RuBP is regenerated. "
        "Fixing one CO2 consumes 3 ATP and 2 NADPH. Our lab, experiment B: leaf discs kept in "
        "the dark beaker never floated, even after 30 minutes."
    ),
}
RAG_QUESTION = ("Using my notes and the attached file, explain how the two stages of photosynthesis "
                "work together, and what our two leaf-disc experiments showed.")
# Planted facts that cannot come from general knowledge — the reliable G2 signal.
# (Topic terms are recorded too, but the model can produce them unaided.)
RAG_MARKERS = {
    "vault": ["7 minutes", "seven minutes", "led lamp", "bicarbonate"],
    "attached": ["dark beaker", "30 minutes", "thirty minutes", "never floated"],
}

PROBES: dict[str, dict] = {
    # (a) greeting with deep-dive + expert: must stay 1–2 sentences.
    # In English an exact "hi" is answered by the router's canned reply, so the
    # English probe uses a non-canned greeting to actually exercise the LLM.
    "a1_greeting_en": {**_c(True, 8, "CBSE", "en", "deep-dive", "expert", "professor"),
                       "question": "hello there!", "kind": "greeting"},
    "a2_greeting_ta": {**_c(True, 8, "CBSE", "ta", "deep-dive", "expert", "professor"),
                       "question": "hi", "kind": "greeting"},
    # (b) off-subject question with a subject set: correct but brief.
    "b_off_subject": {**_c(True, 8, "CBSE", "en", "detailed", "medium", "mentor", subject="Mathematics"),
                      "question": "Who wrote the Ramayana?", "kind": "off_subject"},
    # (c) Grade 2 asks an out-of-scope (calculus) question.
    "c_g2_derivative": {**_c(True, 2, "CBSE", "en", "summary", "medium", "mentor"),
                        "question": "What is a derivative?", "kind": "out_of_scope"},
    # (d) RAG: attached file + Vault doc together (G2), answer at learner level.
    "d1_rag_g5": {**_c(True, 5, "CBSE", "en", "summary", "medium", "mentor"),
                  "question": RAG_QUESTION, "kind": "rag"},
    "d2_rag_g11": {**_c(True, 11, "ICSE", "en", "detailed", "hard", "professor"),
                   "question": RAG_QUESTION, "kind": "rag"},
    # (e) requested format beats board framing (pre-Phase-4 precedence item 4):
    # combo G — expected a list of exam-style questions with short answers/hints, in Tamil.
    "g_format_important_questions": {**_c(True, 10, "Tamil Nadu State Board", "ta", "detailed", "medium", "mentor",
                                         subject="Science"),
                                     "question": "important questions on light reflection", "kind": "format"},
}

SCOPE_TABLE = """Syllabus scope reference (best knowledge; confidence in brackets):
- Photosynthesis idea + word equation: CBSE/NCERT Class 7 (2024 'Curiosity' text may shift chapter); ICSE Class 6-7 [medium]
- Photosynthesis balanced chemical equation: CBSE Class 10 'Life Processes'; ICSE Class 10 Biology [high]
- Newton's second law F = ma: CBSE Class 9 'Force and Laws of Motion'; ICSE Class 9; IGCSE Physics [high]
- Quadratic equations by factorisation: CBSE Class 10 Ch.4; ICSE Class 10; IGCSE (Gr 9-10) Extended; factorisation appears earlier in CBSE Class 9 polynomials [high]
- Calculus (limits/derivatives): CBSE/ISC/State Class 11; IB DP; Cambridge AS (IGCSE Additional Maths has basic differentiation) [high / medium for IGCSE Add-Maths]
Board stages: CBSE=NCERT; ICSE (ISC for 11-12); State board (SSLC at 10, HSC at 11-12); IB PYP 1-5, MYP 6-10, DP 11-12; Cambridge Primary 1-6, Lower Secondary 7-8, IGCSE 9-10, AS/A Level 11-12."""

# Model prices, USD per 1M tokens (input, output). Edit to match your contract.
# A model that is not listed is reported as "price unknown" — its cost is never
# estimated or borrowed from another model.
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4-20250514": (3.00, 15.00),
}

JUDGE_DIMS = ["grade_appropriateness", "board_alignment", "difficulty_fit",
              "language_purity", "latex_validity", "personality_fit"]

LANG_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil", "ar": "Arabic"}
LANG_SCRIPT = {"en": "LATIN", "hi": "DEVANAGARI", "ta": "TAMIL", "ar": "ARABIC"}


def build_cases(combo_ids: list[str], probe_ids: list[str]) -> list[dict]:
    cases = []
    for cid in combo_ids:
        for qid, q in QUESTIONS.items():
            cases.append({"case_id": f"c{cid}_{qid}", "combo": cid, "question_id": qid, "question": q,
                          "kind": "greeting" if qid == "greeting" else "standard",
                          "context": COMBOS[cid]["context"], "settings": COMBOS[cid]["settings"]})
    for pid in probe_ids:
        p = PROBES[pid]
        cases.append({"case_id": f"p_{pid}", "combo": f"probe:{pid}", "question_id": pid,
                      "question": p["question"], "kind": p["kind"],
                      "context": p["context"], "settings": p["settings"]})
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Automatic checks
# ─────────────────────────────────────────────────────────────────────────────

_FENCE = re.compile(r"```.*?```", re.S)
_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
_INLINE = re.compile(r"(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$(?!\$)", re.S)


def strip_math_and_code(text: str) -> str:
    text = _FENCE.sub(" ", text)
    text = _DISPLAY.sub(" MATH ", text)
    return _INLINE.sub(" MATH ", text)


def math_segments(text: str) -> list[str]:
    text = _FENCE.sub(" ", text)
    segs = [m.strip() for m in _DISPLAY.findall(text)]
    segs += [m.strip() for m in _INLINE.findall(_DISPLAY.sub(" ", text))]
    return [s for s in segs if s]


def word_count(text: str) -> int:
    prose = strip_math_and_code(text)
    return sum(1 for tok in re.findall(r"\S+", prose) if re.search(r"\w", tok))


def sentence_count(text: str) -> int:
    prose = strip_math_and_code(text).strip()
    parts = [p for p in re.split(r"(?<=[.!?।؟])\s+|\n+", prose) if re.search(r"\w", p)]
    return len(parts)


def script_purity(text: str, language: str) -> float | None:
    """Share of letters (outside math/code) written in the target script."""
    target = LANG_SCRIPT.get(language)
    if not target:
        return None
    counts = {"target": 0, "other": 0}
    for ch in strip_math_and_code(text):
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        counts["target" if name.startswith(target) else "other"] += 1
    total = counts["target"] + counts["other"]
    return round(counts["target"] / total, 3) if total else None


def length_score(words: int, lo: int, hi: int) -> int:
    if lo <= words <= hi:
        return 5
    if lo * (1 - LENGTH_TOLERANCE) <= words <= hi * (1 + LENGTH_TOLERANCE):
        return 4
    ratio = words / hi if words > hi else (lo / max(words, 1))
    return 3 if ratio <= 1.5 else 2 if ratio <= 2.5 else 1


def length_eval(case: dict, words: int, sentences: int) -> dict:
    """Length target + score for a case. Greetings: sentence cap. Off-subject: brief
    (no more than the summary target, precedence item 3). Otherwise the matrix target."""
    ctx, cs = case["context"], case["settings"]
    lang = ctx.get("language") or "en"
    band = grade_band(ctx.get("grade")) if cs["student_mode"] else None
    if case["kind"] == "greeting":
        return {"length_target": f"<= {GREETING_MAX_SENTENCES} sentences", "length_score": greeting_score(sentences)}
    if case["kind"] == "off_subject":
        _, hi = target_words(band, "summary", lang)
        return {"length_target": [0, hi], "length_score": length_score(words, 0, hi)}
    lo, hi = target_words(band, cs["content_length"], lang)
    return {"length_target": [lo, hi], "length_score": length_score(words, lo, hi)}


def greeting_score(sentences: int) -> int:
    return 5 if sentences <= GREETING_MAX_SENTENCES else 3 if sentences == GREETING_MAX_SENTENCES + 1 else 1


_ITEM_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*•]|\*\*\d+[.)])\s+", re.M)


def question_items(text: str) -> int:
    """Numbered/bulleted lines that are questions (contain '?' or start a numbered item)."""
    lines = [l for l in text.splitlines() if _ITEM_LINE.match(l)]
    return sum(1 for l in lines if "?" in l or re.match(r"^\s*(?:\*\*)?\d+[.)]", l))


def rag_markers(text: str) -> dict:
    low = text.lower()
    return {src: sorted({m for m in marks if m in low}) for src, marks in RAG_MARKERS.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def router_messages(case: dict) -> tuple[list[dict], bool]:
    """Messages exactly as the router would hand them to stream_chat."""
    q = case["question"]
    messages = [{"role": "user", "content": q}]
    if case["kind"] != "rag":
        return messages, False
    lang = case["context"].get("language")
    # Router order (ai_assistant._prepare_chat_turn): vault RAG, then inline docs.
    messages = _inject_rag(messages, q, VAULT_EXCERPT, language=lang)
    messages = _inject_inline_docs(messages, q, [ATTACHED_DOC], language=lang)
    return messages, True


async def generate(case: dict) -> dict:
    lang = case["context"].get("language")
    canned_ok = not lang or lang.lower() in ("en", "english")
    canned = _get_canned_response(case["question"]) if canned_ok else None
    if canned:
        return {"answer": canned, "canned": True, "generations": [], "latency_s": 0.0, "error": None}

    messages, has_files = router_messages(case)
    metas: list[dict] = []
    token = ai_mod.generation_observer.set(metas.append)
    t0 = time.monotonic()
    answer, error = "", None
    try:
        ai = ai_mod.AIService()
        async for chunk in ai.stream_chat(messages=messages, context=dict(case["context"]),
                                          chat_settings=dict(case["settings"]), has_files=has_files):
            answer += chunk
    except Exception as exc:  # recorded, not raised — one failure must not stop the run
        error = f"{type(exc).__name__}: {exc}"
    finally:
        ai_mod.generation_observer.reset(token)
    return {"answer": answer, "canned": False, "generations": metas,
            "latency_s": round(time.monotonic() - t0, 2), "error": error}


# ─────────────────────────────────────────────────────────────────────────────
# Judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a strict, experienced school teacher and curriculum reviewer grading an AI tutor's answer "
    "against the exact settings it was given. Score each dimension 1-5 (5 = fully meets the setting, "
    "3 = noticeable problems, 1 = clearly wrong for the setting). Use null when a dimension does not "
    "apply (e.g. board_alignment when no board applies or student mode is off; latex_validity when the "
    "answer contains no math). Judge correctness too: factual or mathematical errors lower "
    "grade_appropriateness and difficulty_fit. Reply with ONLY a JSON object."
)


def judge_prompt(case: dict, answer: str) -> str:
    ctx, cs = case["context"], case["settings"]
    sm = cs["student_mode"]
    learner = (f"Student mode ON. Learner: Grade {ctx.get('grade')}, board {ctx.get('board')}, "
               f"subject {ctx.get('subject') or 'not set'}.") if sm else \
        "Student mode OFF (general user): grade/board must NOT shape the answer; difficulty is absolute."
    extra = ""
    if case["kind"] == "greeting":
        extra = "This is a greeting/small talk: the correct answer is 1-2 short sentences regardless of the length setting."
    elif case["kind"] == "off_subject":
        extra = (f"The question is outside the set subject ({ctx.get('subject')}). Correct behaviour: answer "
                 "correctly but briefly, without forcing a link to the subject. Add field \"kept_brief\": true/false.")
    elif case["kind"] == "out_of_scope":
        extra = ("The question is beyond the learner's grade. Correct behaviour: still answer correctly, at the "
                 "learner's vocabulary and concept level, without refusing and without mentioning the grade. "
                 "Add fields \"refused\": true/false and \"age_appropriate_explanation\": true/false.")
    elif case["kind"] == "format":
        extra = ("The user explicitly asked for a FORMAT: a list of important / exam-style questions. Correct "
                 "behaviour: produce that list (each question with a short answer or an answer hint), NOT a "
                 "definition → explanation essay; still in the response language, at the learner's level and "
                 "syllabus. Add fields \"produced_requested_format\": true/false and "
                 "\"has_answers_or_hints\": true/false.")
    elif case["kind"] == "rag":
        extra = ("The AI was given two sources: VAULT NOTES about the light-dependent reactions (photosystem II/I, "
                 "photolysis, ATP synthase, NADPH; experiment A: discs floated after 7 minutes under an LED lamp) and an ATTACHED FILE "
                 "about the Calvin cycle (Rubisco, RuBP, G3P; experiment B: discs in the dark beaker never floated). "
                 "Both sources were written above the learner's level. Correct behaviour: use BOTH sources and pitch "
                 "the explanation at the learner's level, not the documents' level. Add fields "
                 "\"used_vault_source\": true/false, \"used_attached_source\": true/false, "
                 "\"pitched_at_learner_level\": true/false.")
    return f"""SETTINGS THE AI WAS GIVEN
{learner}
Response language: {LANG_NAMES.get(ctx.get('language'), ctx.get('language'))}
Difficulty: {cs['difficulty']} ({'relative to the learner grade' if sm else 'absolute level'})
Content length: {cs['content_length']}
Personality: {cs['personality']}
Explain in 3 Ways: {'ON (expects Analogy / Technical / Real-World Example sections)' if cs.get('explain_3ways') else 'off'}
Examples setting: {'ON' if cs.get('examples') else 'off'}
{extra}

{SCOPE_TABLE}

Rules the answer must also follow: never mention the learner's grade or board; math in $...$/$$...$$ LaTeX; no emojis expected.

QUESTION
{case['question']}

ANSWER TO GRADE
<<<
{answer}
>>>

Return JSON: {{"grade_appropriateness": int|null, "board_alignment": int|null, "difficulty_fit": int|null,
"language_purity": int|null, "latex_validity": int|null, "personality_fit": int|null,
"mentions_grade_or_board": true|false, "notes": "<=40 words on the main problems"}} plus any extra fields requested above."""


async def judge(client, case: dict, answer: str, model: str) -> dict:
    t0 = time.monotonic()
    resp = await client.messages.create(
        # No `temperature`: it is deprecated for current Claude models (400 error).
        model=model, max_tokens=600,
        system=JUDGE_SYSTEM, messages=[{"role": "user", "content": judge_prompt(case, answer)}],
    )
    text = resp.content[0].text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {"notes": f"judge returned unparseable JSON: {text[:200]}"}
    data["_usage"] = {"model": model, "input_tokens": resp.usage.input_tokens,
                      "output_tokens": resp.usage.output_tokens, "latency_s": round(time.monotonic() - t0, 2)}
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_case(case: dict, judge_client, judge_model: str, sem: asyncio.Semaphore) -> dict:
    async with sem:
        gen = await generate(case)
        ctx, cs = case["context"], case["settings"]
        lang = ctx.get("language") or "en"
        band = grade_band(ctx.get("grade")) if cs["student_mode"] else None
        answer = gen["answer"]

        auto: dict = {"words": word_count(answer), "sentences": sentence_count(answer),
                      "script_purity": script_purity(answer, lang)}
        auto.update(length_eval(case, auto["words"], auto["sentences"]))
        segs = math_segments(answer)
        auto["math_segments"] = len(segs)
        if segs:
            errors = await katex_check.check_segments(segs)
            auto["katex_errors"] = None if errors is None else len(errors)
            auto["katex_error_samples"] = list((errors or {}).values())[:3]
        if case["kind"] == "rag":
            auto["rag_markers"] = rag_markers(answer)
        if case["kind"] == "format":
            auto["question_items"] = question_items(answer)

        verdict = {}
        if gen["canned"]:
            verdict = {"notes": "router canned reply (no LLM)"}
        elif answer and judge_client is not None:
            try:
                verdict = await judge(judge_client, case, answer, judge_model)
            except Exception as exc:
                verdict = {"notes": f"judge failed: {type(exc).__name__}: {exc}"}
        record = {**case, **gen, "auto": auto, "judge": verdict}
        print(f"  done {case['case_id']:<22} words={auto['words']:<5} len={auto['length_score']} "
              f"finish={[g.get('finish_reason') for g in gen['generations']]} {'ERR ' + gen['error'] if gen['error'] else ''}",
              flush=True)
        return record


async def run(args) -> Path:
    combos = [c for c in args.combos.split(",") if c] if args.combos else []
    probes = list(PROBES) if args.probes == "all" else [p for p in (args.probes or "").split(",") if p]
    cases = build_cases(combos, probes)
    out = Path(args.out)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"Running {len(cases)} cases → {out}", flush=True)

    import anthropic
    judge_client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY) if settings.ANTHROPIC_API_KEY else None
    sem = asyncio.Semaphore(args.concurrency)

    # --force-provider: hide the other generators from AIService for this run
    # (the judge uses its own client and is unaffected).
    getters = {"gemini": "_get_gemini_async", "claude": "_get_anthropic", "openai": "_get_openai"}
    saved = {name: getattr(ai_mod.AIService, name) for name in getters.values()}
    if args.force_provider:
        for prov, name in getters.items():
            if prov != args.force_provider:
                setattr(ai_mod.AIService, name, lambda self: None)
    saved_guard = settings.CONTENT_GUARD_LLM_CLASSIFIER
    if args.guard_llm != "auto":
        settings.CONTENT_GUARD_LLM_CLASSIFIER = args.guard_llm == "on"
    guard_state = settings.CONTENT_GUARD_LLM_CLASSIFIER
    try:
        records = await asyncio.gather(*(evaluate_case(c, judge_client, args.judge_model, sem) for c in cases))
    finally:
        for name, fn in saved.items():
            setattr(ai_mod.AIService, name, fn)
        settings.CONTENT_GUARD_LLM_CLASSIFIER = saved_guard

    with open(out / "raw.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in records:
        (out / "outputs" / f"{r['case_id']}.md").write_text(
            f"# {r['case_id']}\n\n**Question:** {r['question']}\n\n**Context:** `{json.dumps(r['context'], ensure_ascii=False)}`\n\n"
            f"**Settings:** `{json.dumps(r['settings'])}`\n\n---\n\n{r['answer']}\n"
        )
    meta = {"run_at": datetime.now(timezone.utc).isoformat(), "primary_model": settings.AI_PRIMARY_MODEL,
            "document_model": settings.AI_DOCUMENT_MODEL, "fallback_model": settings.AI_FALLBACK_MODEL,
            "judge_model": args.judge_model,
            "guard_llm_classifier": guard_state, "forced_provider": args.force_provider,
            "compare_to": args.compare, "combos": combos, "probes": probes}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    write_report(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def probe_result(r: dict) -> tuple[str, str, str]:
    """(expectation, PASS/FAIL/PARTIAL, detail) for one probe record."""
    j, a = r["judge"], r["auto"]
    if r["kind"] == "greeting":
        ok = a["sentences"] <= GREETING_MAX_SENTENCES
        return ("1–2 sentences despite deep-dive + expert", "PASS" if ok else "FAIL",
                f"{a['sentences']} sentence(s), {a['words']} words")
    if r["kind"] == "off_subject":
        ok = j.get("kept_brief") is True
        return ("correct but brief", "PASS" if ok else "FAIL", f"{a['words']} words; judge kept_brief={j.get('kept_brief')}")
    if r["kind"] == "out_of_scope":
        ok = j.get("refused") is False and j.get("age_appropriate_explanation") is True
        return ("answers simply, no refusal, no grade mention", "PASS" if ok else "FAIL",
                f"refused={j.get('refused')}, age_appropriate={j.get('age_appropriate_explanation')}, "
                f"grade score {j.get('grade_appropriateness')}")
    if r["kind"] == "format":
        purity = a.get("script_purity")
        ok = (j.get("produced_requested_format") is True and j.get("has_answers_or_hints") is True
              and a.get("question_items", 0) >= 3 and (purity or 0) >= 0.9)
        return ("list of exam-style questions with short answers/hints, in Tamil", "PASS" if ok else "FAIL",
                f"{a.get('question_items')} question items; judge format={j.get('produced_requested_format')}, "
                f"answers/hints={j.get('has_answers_or_hints')}; script purity {purity}")
    mk = a.get("rag_markers", {})
    both = bool(mk.get("vault")) and bool(mk.get("attached"))
    level = j.get("pitched_at_learner_level") is True
    # G2 uses the planted-fact markers: the judge cannot tell a source's
    # content from the model's own general knowledge.
    verdict = "PASS" if both and level else "PARTIAL" if both or level else "FAIL"
    return ("uses BOTH sources (G2) at the learner's level", verdict,
            f"G2 {'PASS' if both else 'FAIL'} (planted facts — vault {mk.get('vault')}, attached {mk.get('attached')}); "
            f"learner level (judge): {j.get('pitched_at_learner_level')}")


def _dims_of(records: list[dict]) -> dict:
    llm = [r for r in records if not r["canned"]]
    dims = {d: _avg([r["judge"].get(d) for r in llm]) for d in JUDGE_DIMS}
    dims["length_compliance (auto)"] = _avg([r["auto"]["length_score"] for r in llm])
    return dims


def _case_mean(r: dict) -> float | None:
    vals = [r["judge"].get(d) for d in JUDGE_DIMS] + [r["auto"]["length_score"]]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(statistics.mean(vals), 2) if vals else None


def compare_section(base_dir: Path, records: list[dict], dims: dict) -> list[str]:
    base = load_records(base_dir)
    base_meta = json.loads((base_dir / "meta.json").read_text())
    bdims = _dims_of(base)
    out = ["", f"## Before / after — vs `{base_dir.name}`", "",
           f"Baseline generator mix: {', '.join(f'{k} × {v}' for k, v in sorted(_provider_mix([r for r in base if not r['canned']]).items()))}; "
           f"judge `{base_meta.get('judge_model')}`.", "",
           "| Dimension | Before | After | Δ |", "|---|---|---|---|"]
    for d in dims:
        b, a = bdims.get(d), dims.get(d)
        delta = f"{a - b:+.2f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "–"
        out.append(f"| {d} | {b} | {a} | {delta} |")
    by_id = {r["case_id"]: r for r in base}
    probes = [r for r in records if r["combo"].startswith("probe:")]
    if probes:
        out += ["", "| Probe | Before | After | After detail |", "|---|---|---|---|"]
        for r in probes:
            b = by_id.get(r["case_id"])
            out.append(f"| {r['case_id']} | {probe_result(b)[1] if b else '–'} | **{probe_result(r)[1]}** | {probe_result(r)[2]} |")
    out += ["", "| Case | Before mean | After mean | Before words | After words | Length target |", "|---|---|---|---|---|---|"]
    for r in records:
        b = by_id.get(r["case_id"])
        if r["canned"] or b is None:
            continue
        out.append(f"| {r['case_id']} | {_case_mean(b)} | {_case_mean(r)} | {b['auto']['words']} | {r['auto']['words']} | {r['auto']['length_target']} |")
    return out


def _fmt_cost(cost: float | None) -> str:
    return "price unknown" if cost is None else f"{cost:.4f}"


def cost_breakdown(records: list[dict], judge_model: str) -> tuple[list[dict], float, list[str]]:
    """Per-model token totals and cost. Unknown prices → cost None ("price unknown")."""
    usage: dict[tuple[str, str], dict] = {}

    def add(role, model, tin, tout):
        row = usage.setdefault((role, model), {"role": role, "model": model, "input": 0, "output": 0, "calls": 0})
        row["input"] += tin or 0
        row["output"] += tout or 0
        row["calls"] += 1

    for r in records:
        for g in r.get("generations", []):
            add("generation", g.get("model") or "?", g.get("input_tokens"),
                (g.get("output_tokens") or 0) + (g.get("thoughts_tokens") or 0))
        u = (r.get("judge") or {}).get("_usage")
        if u:
            add("judge", u.get("model") or judge_model, u.get("input_tokens"), u.get("output_tokens"))

    rows, total, unknown = [], 0.0, []
    for row in usage.values():
        price = MODEL_PRICES_USD_PER_MTOK.get(row["model"])
        if price is None:
            row["cost"] = None
            unknown.append(row["model"])
        else:
            row["cost"] = (row["input"] * price[0] + row["output"] * price[1]) / 1e6
            total += row["cost"]
        rows.append(row)
    return rows, total, sorted(set(unknown))


def _provider_mix(records: list[dict]) -> dict[str, int]:
    mix: dict[str, int] = {}
    for r in records:
        key = "/".join(f"{g.get('provider')}:{g.get('model')}" for g in r["generations"]) or "none (error/empty)"
        mix[key] = mix.get(key, 0) + 1
    return mix


def _avg(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.mean(xs), 2) if xs else None


def load_records(run_dir: Path) -> list[dict]:
    """Load a run; automatic length scores are recomputed so every run is scored by the current rules."""
    records = [json.loads(line) for line in open(run_dir / "raw.jsonl")]
    for r in records:
        r["auto"].update(length_eval(r, r["auto"]["words"], r["auto"]["sentences"]))
    return records


def write_report(out: Path) -> None:
    records = load_records(out)
    meta = json.loads((out / "meta.json").read_text())
    llm = [r for r in records if not r["canned"]]

    dims = {d: _avg([r["judge"].get(d) for r in llm]) for d in JUDGE_DIMS}
    dims["length_compliance (auto)"] = _avg([r["auto"]["length_score"] for r in records if not r["canned"]])
    purity = [r["auto"]["script_purity"] for r in llm if r["auto"]["script_purity"] is not None]

    # tokens & cost
    finish_rows = [(r["case_id"], g.get("provider"), g.get("model"), g.get("output_tokens"), g.get("finish_reason"))
                   for r in llm for g in r["generations"]]
    cost_rows, known_total, unknown_models = cost_breakdown(llm, meta["judge_model"])
    truncated = [row for row in finish_rows if str(row[4]).upper() not in ("STOP", "FINISHREASON.STOP", "END_TURN", "STOP_SEQUENCE", "NONE")]
    missing_meta = [r["case_id"] for r in llm if not r["generations"]]

    def score_of(r):
        vals = [r["judge"].get(d) for d in JUDGE_DIMS] + [r["auto"]["length_score"]]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return statistics.mean(vals) if vals else 0

    lines = [f"# Tailoring eval — {out.name}", "",
             f"Run at {meta['run_at']} · generator routing: primary `{meta['primary_model']}`, documents `{meta['document_model']}`, "
             f"fallback `{meta['fallback_model']}` · forced generator: {meta.get('forced_provider') or 'none (default routing)'} · "
             f"judge `{meta['judge_model']}` · guard LLM classifier: {meta['guard_llm_classifier']}",
             f"Cases: {len(records)} ({len(llm)} LLM generations, {len(records) - len(llm)} router canned replies). "
             f"Combos {', '.join(meta['combos']) or '—'}; probes {', '.join(meta['probes']) or '—'}.", "",
             "**Provider mix (who actually answered):** "
             + ", ".join(f"{k} × {v}" for k, v in sorted(_provider_mix(llm).items())), "",
             "## Per-dimension averages (1–5; LLM generations only)", "", "| Dimension | Avg | Target |", "|---|---|---|"]
    for d, v in dims.items():
        lines.append(f"| {d} | {v if v is not None else 'n/a'} | ≥ 4.0 |")
    lines += ["", f"Automatic script purity (share of letters in the target script): mean "
              f"{_avg(purity)}, min {min(purity) if purity else 'n/a'}.", ""]

    lines += ["## Cost & tokens", "", "| Role | Model | Calls | Input tokens | Output tokens | Cost (USD) |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {r['role']} | {r['model']} | {r['calls']} | {r['input']:,} | {r['output']:,} | "
              f"{_fmt_cost(r['cost'])} |" for r in cost_rows]
    lines += ["", f"**Total for priced models: ${known_total:.4f}**"
              + (f" — plus {', '.join(unknown_models)} (price unknown; add it to `MODEL_PRICES_USD_PER_MTOK`)." if unknown_models else "."),
              "Excludes content-guard classifier calls (separate client). Prices come from `MODEL_PRICES_USD_PER_MTOK`.", ""]

    lines += ["## Truncation / finish_reason", ""]
    lines.append("No truncation: every generation finished normally." if not truncated else
                 "**Generations that did not finish normally:** " + "; ".join(f"{c} ({p}, {f}, {o} tok)" for c, p, m, o, f in truncated))
    if missing_meta:
        lines.append(f"\nNo provider metadata captured for: {', '.join(missing_meta)} (error or empty stream).")
    lines += ["", "| Case | Provider | Model | Output tokens | finish_reason |", "|---|---|---|---|---|"]
    lines += [f"| {c} | {p} | {m} | {o} | {f} |" for c, p, m, o, f in finish_rows]

    lines += ["", "## Per-case scores", "",
              "| Case | Lang | Len target | Words | Len | Grade | Board | Diff | Lang | LaTeX | Pers | KaTeX err | Notes |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        j, a = r["judge"], r["auto"]
        lines.append(
            f"| {r['case_id']} | {r['context'].get('language')} | {a['length_target']} | {a['words']} | {a['length_score']} | "
            + " | ".join(str(j.get(d, '–')) for d in JUDGE_DIMS)
            + f" | {a.get('katex_errors', '–')} | {(j.get('notes') or '').replace('|', '/')[:140]} |")

    lowest = sorted(llm, key=score_of)[:10]
    lines += ["", "## Lowest-scoring cases", ""]
    for r in lowest:
        lines += [f"### {r['case_id']} — mean {score_of(r):.2f}",
                  f"Settings: `{json.dumps(r['settings'])}` · context: `{json.dumps(r['context'], ensure_ascii=False)}`",
                  f"Judge: {r['judge'].get('notes', '')}", "", "> " + r["answer"][:600].replace("\n", "\n> ") + ("…" if len(r["answer"]) > 600 else ""), ""]

    probes = [r for r in records if r["combo"].startswith("probe:")]
    if probes:
        lines += ["## Probes", "", "| Probe | Expectation | Verdict | Detail |", "|---|---|---|---|"]
        for r in probes:
            exp, verdict, detail = probe_result(r)
            lines.append(f"| {r['case_id']} | {exp} | **{verdict}** | {detail} |")

    if meta.get("compare_to"):
        lines += compare_section(Path(meta["compare_to"]), records, dims)

    # Language calibration (amendment 4): words per equivalent answer, by language.
    by_lang: dict[str, list[float]] = {}
    for r in llm:
        if r["kind"] != "standard":
            continue
        lo, hi = r["auto"]["length_target"]
        mid = (lo + hi) / 2
        lang = r["context"].get("language") or "en"
        by_lang.setdefault(lang, []).append(r["auto"]["words"] / mid if mid else 0)
    lines += ["", "## Language calibration (amendment 4)", "",
              "Actual words ÷ midpoint of the (multiplier-adjusted) target, per language. Values near 1.0 mean the "
              "multiplier is right; a consistent offset across languages means the prompt, not the multiplier, is off.", "",
              "| Language | n | mean ratio |", "|---|---|---|"]
    lines += [f"| {lng} | {len(v)} | {round(statistics.mean(v), 2)} |" for lng, v in sorted(by_lang.items())]

    tok_per_word: dict[str, list[float]] = {}
    for r in llm:
        outs = sum(g.get("output_tokens") or 0 for g in r["generations"])
        if outs and r["auto"]["words"] >= 20:
            tok_per_word.setdefault(r["context"].get("language") or "en", []).append(outs / r["auto"]["words"])
    lines += ["", "## Measured output tokens per word (amendment 5)", "", "| Language | n | mean | max |", "|---|---|---|---|"]
    lines += [f"| {lng} | {len(v)} | {round(statistics.mean(v), 2)} | {round(max(v), 2)} |" for lng, v in sorted(tok_per_word.items())]

    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nREPORT → {out / 'REPORT.md'}")
    print("Per-dimension averages:", json.dumps(dims))
    print(f"Cost (priced models) ${known_total:.4f}; price unknown: {', '.join(unknown_models) or 'none'}")


async def rejudge(out: Path, judge_model: str, concurrency: int) -> None:
    """Re-score an existing run's answers with the judge (no regeneration)."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    records = [json.loads(line) for line in open(out / "raw.jsonl")]
    sem = asyncio.Semaphore(concurrency)

    async def one(r):
        if r["canned"] or not r["answer"]:
            return r
        async with sem:
            try:
                r["judge"] = await judge(client, r, r["answer"], judge_model)
            except Exception as exc:
                r["judge"] = {"notes": f"judge failed: {type(exc).__name__}: {exc}"}
        print(f"  judged {r['case_id']}", flush=True)
        return r

    records = await asyncio.gather(*(one(r) for r in records))
    with open(out / "raw.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = json.loads((out / "meta.json").read_text())
    meta["judge_model"] = judge_model
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    write_report(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--combos", default="1,4,7,12,13,18")
    p.add_argument("--probes", default="all", help="'all' or comma-separated probe ids")
    p.add_argument("--out", default=f"eval_results/run_{datetime.now():%Y%m%d_%H%M%S}")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--judge-model", default="claude-sonnet-5-5",
                   help="Claude model for the judge (AI_DOCUMENT_MODEL may be unavailable)")
    p.add_argument("--report-only", help="rebuild REPORT.md from an existing run directory")
    p.add_argument("--force-provider", choices=["gemini", "claude", "openai"],
                   help="use only this generator (others hidden) — for like-for-like comparisons")
    p.add_argument("--guard-llm", choices=["auto", "on", "off"], default="auto",
                   help="content-guard LLM classifier for this run (auto = .env setting)")
    p.add_argument("--compare", help="earlier run directory to compare against in REPORT.md")
    p.add_argument("--rejudge", help="re-run only the judge on an existing run directory")
    args = p.parse_args()
    if args.rejudge:
        asyncio.run(rejudge(Path(args.rejudge), args.judge_model, args.concurrency))
        return
    if args.report_only:
        write_report(Path(args.report_only))
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
