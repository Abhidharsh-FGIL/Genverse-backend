"""AI Assistant tailoring verification over the REAL local HTTP path.

Drives a LOCAL backend exactly as the frontend does — PUT /users/me/context,
create a fresh chat, PATCH its settings, GET them back, POST
/chats/{id}/messages/stream and parse the SSE stream — across one-factor
sweeps S1–S12, then scores every answer:

  * automatic checks: length vs the matrix (±25 %, per language), sentence
    length vs band limits, structure counts, 3-ways labels, exam notes,
    personality ending, script purity, English-in-brackets rule, never naming
    grade/board, LaTeX (real KaTeX), no \\( \\[, no $-currency, no emojis,
    truncation heuristics, numeric correctness;
  * differentiation: grade trends (Spearman), length monotonicity, blind
    difficulty ordering, pairwise similarity, blind identification of
    grade / board / personality / length with confusion matrices;
  * an LLM judge rubric (settings visible).

Provider/model per answer comes from the backend's own "[MODEL] …" log lines,
so the backend must run with INFO logging for `app` (see --backend-log) and
cases run one at a time. finish_reason is NOT exposed over HTTP: truncation
is detected heuristically (unclosed math/code, no terminal punctuation).

SAFETY: --base-url must be localhost/127.0.0.1. Never point this at a server.

    .venv/bin/python scripts/tailoring_verify.py --quick --dry-run
    .venv/bin/python scripts/tailoring_verify.py --quick --backend-log /path/backend.log
"""
import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import settings  # noqa: E402
from app.prompts.matrices import GREETING_MAX_SENTENCES, LENGTH_TOLERANCE, grade_band, target_words  # noqa: E402
from app.services import katex_check  # noqa: E402
from eval_tailoring import (  # noqa: E402
    MODEL_PRICES_USD_PER_MTOK, math_segments, sentence_count, strip_math_and_code, word_count,
)

BANK_PATH = ROOT / "tests" / "tailoring" / "questions.yaml"
BOARDS_ALL = ["CBSE", "ICSE", "Tamil Nadu State Board", "State Board", "IB", "Cambridge", None]
BOARDS_BLIND = ["CBSE", "ICSE", "State Board", "IB", "Cambridge"]
PERSONALITIES = ["mentor", "coach", "professor", "friend", "technical-expert"]
LENGTHS = ["small", "summary", "detailed", "deep-dive"]
DIFFICULTIES = ["easy", "medium", "hard", "expert"]
LANGS = ["en", "hi", "ta", "ar"]
SENTENCE_LIMIT = {"early-primary": 12, "upper-primary": 15, "middle-school": 20, "high-school": 25}
SCRIPT_PREFIX = {"en": "LATIN", "hi": "DEVANAGARI", "ta": "TAMIL", "ar": "ARABIC"}
LANG_NAME = {"en": "English", "hi": "Hindi", "ta": "Tamil", "ar": "Arabic"}

# ─────────────────────────────────────────────────────────────────────────────
# Cases
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Case:
    case_id: str
    sweep: str
    question: str
    qkind: str                       # concept | problem | format | greeting | off_subject | rag | midchat
    grade: Optional[int] = 8
    board: Optional[str] = "CBSE"
    lang: str = "en"
    sm: bool = True
    length: str = "summary"
    difficulty: str = "medium"
    personality: str = "mentor"
    subject: Optional[str] = None
    three_ways: bool = False
    examples: bool = False
    answers: Optional[list] = None
    fmt: Optional[str] = None
    group: Optional[str] = None      # blind-test / trend group key
    factor: Optional[str] = None     # the swept value, for grouping
    sample: int = 0
    enhancements: bool = False
    midchat: Optional[list] = None   # [(question, three_ways)] for S7 mid-chat toggle


def load_bank() -> dict:
    return yaml.safe_load(BANK_PATH.read_text())


def problem_for(bank: dict, grade: Optional[int]) -> dict:
    g = grade or 10
    for subj in ("maths", "science", "social_science", "general_knowledge"):
        for p in bank["subjects"][subj]["problems"]:
            if p["grades"][0] <= g <= p["grades"][1]:
                return p
    return bank["subjects"]["general_knowledge"]["problems"][0]


def three_questions(bank: dict, grade: Optional[int]) -> list[tuple[str, str, Optional[list]]]:
    p = problem_for(bank, grade)
    return [("concept", bank["subjects"]["science"]["concept"], None),
            ("problem", p["q"], p["answers"]),
            ("concept", bank["subjects"]["social_science"]["concept"], None)]


def build_cases(bank: dict, quick: bool, samples: int) -> list[Case]:
    cases: list[Case] = []

    def add(sweep, factor, cfg, questions, **extra):
        for s in range(samples):
            for qi, (qkind, q, ans) in enumerate(questions):
                cid = f"{sweep}_{factor}_q{qi}_s{s}".replace(" ", "")
                cases.append(Case(case_id=cid, sweep=sweep, question=q, qkind=qkind, answers=ans,
                                  factor=str(factor), sample=s, group=f"{sweep}|q{qi}|s{s}", **{**cfg, **extra}))

    # S1 grade
    for g in ([2, 5, 8, 11] if quick else list(range(1, 13))):
        add("S1", f"g{g}", {"grade": g}, three_questions(bank, g))
    # S2 board × grade
    for g in ([10] if quick else [3, 7, 10, 12]):
        for b in BOARDS_ALL:
            add("S2", f"g{g}_{b or 'none'}", {"grade": g, "board": b}, three_questions(bank, g))
    # S3 length × grade
    for g in ([9] if quick else [2, 6, 9, 12]):
        for ln in LENGTHS:
            add("S3", f"g{g}_{ln}", {"grade": g, "length": ln}, three_questions(bank, g))
    # S4 difficulty × grade (+ student mode off)
    for g in ([8] if quick else [4, 8, 11]):
        for d in DIFFICULTIES:
            add("S4", f"g{g}_{d}", {"grade": g, "difficulty": d}, three_questions(bank, g))
    for d in (["hard"] if quick else DIFFICULTIES):
        add("S4", f"off_{d}", {"grade": 8, "sm": False, "difficulty": d}, three_questions(bank, 8))
    # S5 personality: one factual + one problem-solving question
    for g in ([10] if quick else [5, 10]):
        qs = three_questions(bank, g)[:2]
        for p in PERSONALITIES:
            add("S5", f"g{g}_{p}", {"grade": g, "personality": p}, qs)
    # S6 language
    s6 = [(8, "hi"), (8, "ta"), (8, "ar"), (11, "ta")] if quick else [(g, l) for g in (3, 8, 11) for l in LANGS]
    for g, l in s6:
        subj = "Science" if g == 11 else None   # grade 11 Science tests the English-term-in-brackets rule
        add("S6", f"g{g}_{l}", {"grade": g, "lang": l, "subject": subj}, three_questions(bank, g))
    # S7 toggles + mid-chat 3-ways off
    concept = [("concept", bank["subjects"]["science"]["concept"], None)]
    for g in ([5] if quick else [5, 10]):
        add("S7", f"g{g}_3ways_on", {"grade": g, "three_ways": True}, concept)
        if not quick:
            add("S7", f"g{g}_3ways_off", {"grade": g, "three_ways": False}, concept)
    for g in ([10] if quick else [5, 10]):
        add("S7", f"g{g}_examples_on", {"grade": g, "examples": True}, concept)
        if not quick:
            add("S7", f"g{g}_examples_off", {"grade": g, "examples": False}, concept)
    cases.append(Case(case_id="S7_midchat_3ways_off", sweep="S7", question="(mid-chat)", qkind="midchat", grade=5,
                      three_ways=True, factor="midchat",
                      midchat=[("What is photosynthesis?", True), ("What is respiration in plants?", True),
                               ("What is transpiration?", False)]))
    # S8 student mode off with grade 8 + CBSE stored
    add("S8", "off_g8_cbse", {"grade": 8, "board": "CBSE", "sm": False}, three_questions(bank, 8))
    # S9 subject set vs unset + off-subject
    add("S9", "subject_science", {"grade": 8, "subject": "Science"}, concept)
    add("S9", "subject_unset", {"grade": 8, "subject": None}, concept)
    add("S9", "off_subject", {"grade": 8, "subject": "Mathematics"}, [("off_subject", bank["off_subject"], None)])
    # S10 requested formats
    fmts = list(bank["formats"].items())
    s10 = ([(10, "Tamil Nadu State Board", "ta", fmts[0]), (10, "Tamil Nadu State Board", "ta", fmts[1]),
            (5, "CBSE", "en", fmts[2]), (12, "CBSE", "en", fmts[3]), (10, "CBSE", "en", fmts[4])] if quick else
           [(g, b, l, f) for g in (5, 10, 12) for (b, l) in (("Tamil Nadu State Board", "ta"), ("CBSE", "en")) for f in fmts])
    for g, b, l, (fname, fq) in s10:
        add("S10", f"g{g}_{l}_{fname}", {"grade": g, "board": b, "lang": l, "subject": "Science", "fmt": fname},
            [("format", fq, None)])
    # S11 greetings with deep-dive + expert
    s11 = [("hi", "en"), ("hi", "ta"), ("thanks", "hi"), ("ok", "ar")] if quick else \
        [(gq, l) for gq in bank["greetings"] for l in LANGS]
    for gq, l in s11:
        add("S11", f"{gq}_{l}", {"grade": 8, "lang": l, "length": "deep-dive", "difficulty": "expert"},
            [("greeting", gq, None)])
    # S12 RAG: attached file + Vault doc together
    for g in (5, 11):
        add("S12", f"g{g}", {"grade": g, "length": "summary" if g == 5 else "detailed"},
            [("rag", RAG_QUESTION, None)])
    # Enhancement subset (practice / follow-ups / next steps / mind map)
    enh_ids = {"S1_g2_q0_s0", "S1_g11_q0_s0", "S6_g8_ta_q0_s0", "S10_g10_ta_important_questions_q0_s0"}
    for c in cases:
        c.enhancements = c.case_id in enh_ids
    return cases


# RAG sources with planted facts (same idea as eval_tailoring.py).
VAULT_DOC = ("Class notes — light-dependent reactions. Photosystem II absorbs light and splits water, releasing "
             "oxygen; ATP and NADPH are produced. Our lab, experiment A: spinach leaf discs in bicarbonate "
             "solution floated after 7 minutes under the LED lamp.")
ATTACHED_DOC = {"filename": "calvin_cycle_notes.txt",
                "text": ("The Calvin cycle fixes carbon dioxide in the stroma using the enzyme Rubisco, producing "
                         "glucose with the ATP and NADPH from the light reactions. Our lab, experiment B: leaf discs "
                         "kept in the dark beaker never floated, even after 30 minutes.")}
RAG_QUESTION = ("Using my notes and the attached file, explain how the two stages of photosynthesis work "
                "together, and what our two leaf-disc experiments showed.")
RAG_MARKERS = {"vault": ["7 minutes", "seven minutes", "led lamp", "bicarbonate"],
               "attached": ["dark beaker", "30 minutes", "thirty minutes", "never floated"]}

# ─────────────────────────────────────────────────────────────────────────────
# Local HTTP client (mirrors the frontend)
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_LINE = re.compile(r"\[MODEL\] (\w+) \(([^)]*)\): (\S+)")


class LocalBackend:
    def __init__(self, base_url: str, email: str, password: str, log_path: Optional[Path]):
        host = urlparse(base_url).hostname
        if host not in ("localhost", "127.0.0.1"):
            raise SystemExit(f"Refusing to run against {host!r}: --base-url must be a LOCAL backend.")
        self.client = httpx.AsyncClient(base_url=base_url, timeout=httpx.Timeout(180.0))
        self.email, self.password, self.log_path = email, password, log_path
        self.headers: dict = {}

    async def login(self):
        r = await self.client.post("/api/v1/auth/login", json={"email": self.email, "password": self.password})
        r.raise_for_status()
        self.headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

    def _log_offset(self) -> int:
        return self.log_path.stat().st_size if self.log_path and self.log_path.exists() else 0

    def _log_since(self, offset: int) -> list[str]:
        if not self.log_path or not self.log_path.exists():
            return []
        with open(self.log_path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace").splitlines()

    async def put_context(self, c: Case):
        body = {"workspace_id": "personal", "grade": c.grade, "board": c.board, "subject": c.subject,
                "language": c.lang, "student_mode": c.sm}
        (await self.client.put("/api/v1/users/me/context", headers=self.headers, json=body)).raise_for_status()

    async def new_chat(self, c: Case) -> tuple[str, dict]:
        chat = (await self.client.post("/api/v1/ai/chats", headers=self.headers,
                                       json={"title": "New Chat", "scope": "personal", "org_id": None})).json()
        cid = chat["id"]
        await self.patch_settings(cid, {"student_mode": c.sm, "difficulty": c.difficulty, "personality": c.personality,
                                        "content_length": c.length, "explain_3ways": c.three_ways,
                                        "examples": c.examples, "mind_map": False, "infographic": False,
                                        "video_refs": False, "practice": False, "followup": False,
                                        "next_steps": False})
        return cid, await self.get_settings(cid)

    async def patch_settings(self, chat_id: str, updates: dict):
        (await self.client.patch(f"/api/v1/ai/chats/{chat_id}/settings", headers=self.headers,
                                 json=updates)).raise_for_status()

    async def get_settings(self, chat_id: str) -> dict:
        r = await self.client.get(f"/api/v1/ai/chats/{chat_id}/settings", headers=self.headers)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def chat_context(c: Case, settings_json: dict, extra: Optional[dict] = None) -> dict:
        """Exactly what AskDocsPage sends (student mode → grade/board/subject; no tone/difficulty/output_mode)."""
        sm = settings_json.get("student_mode", c.sm)
        ctx = {"student_mode": sm, "language": c.lang, "selected_files": [], "workspace_type": "personal",
               "org_id": None}
        if sm:
            ctx.update({"grade": c.grade, "board": c.board, "subject": c.subject})
        ctx.update(extra or {})
        return ctx

    async def stream(self, chat_id: str, message: str, context: dict, chat_settings: dict) -> dict:
        offset = self._log_offset()
        t0, text, events, bad = time.monotonic(), "", 0, 0
        body = {"message": message, "context": context, "chat_settings": chat_settings}
        async with self.client.stream("POST", f"/api/v1/ai/chats/{chat_id}/messages/stream",
                                      headers=self.headers, json=body) as resp:
            status = resp.status_code
            buf = ""
            async for chunk in resp.aiter_text():
                buf += chunk
                lines = buf.split("\n")
                buf = lines.pop()
                for line in lines:                      # same parsing as AskDocsPage.handleSend
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw == "[DONE]":
                        continue
                    try:
                        evt = json.loads(raw)
                        if isinstance(evt.get("delta"), str):
                            text += evt["delta"]
                            events += 1
                    except json.JSONDecodeError:
                        bad += 1
        await asyncio.sleep(0.3)  # let the backend flush its log lines
        log = self._log_since(offset)
        models = [{"provider": m.group(1), "route": m.group(2), "model": m.group(3)}
                  for line in log for m in [_MODEL_LINE.search(line)] if m]
        return {"status": status, "answer": text, "sse_events": events, "sse_bad_events": bad,
                "seconds": round(time.monotonic() - t0, 2), "models": models,
                "served_by": models[-1] if models else None,
                "fallback": any("stream failed" in line for line in log) or len(models) > 1,
                "breaker_trips": sum("breaker_open" in line for line in log),
                "guard_fail_open": sum("content_guard.fail_open" in line for line in log),
                "interrupted": any("after partial output" in line for line in log)}

    async def enhancement(self, chat_id: str, path: str, body: dict) -> dict:
        t0 = time.monotonic()
        r = await self.client.post(f"/api/v1/ai/chats/{chat_id}/{path}", headers=self.headers, json=body)
        return {"status": r.status_code, "seconds": round(time.monotonic() - t0, 2),
                "json": r.json() if r.headers.get("content-type", "").startswith("application/json") else None}

    async def upload_vault(self, text: str, filename: str = "light_reactions_notes.txt") -> Optional[str]:
        files = {"file": (filename, text.encode(), "text/plain")}
        r = await self.client.post("/api/v1/library/upload", headers=self.headers, files=files)
        if r.status_code >= 300:
            print(f"  vault upload failed: {r.status_code} {r.text[:200]}")
            return None
        item_id = r.json()["id"]
        for _ in range(90):
            s = (await self.client.get(f"/api/v1/library/{item_id}/status", headers=self.headers)).json()
            if s.get("processing_status") in ("ready", "failed", "error"):
                print(f"  vault item {item_id[:8]} → {s.get('processing_status')}")
                return item_id if s.get("processing_status") == "ready" else None
            await asyncio.sleep(2)
        print("  vault processing timed out")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Automatic checks
# ─────────────────────────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?।؟])\s+|\n+")
_HEADING = re.compile(r"^\s*#{1,6}\s", re.M)
_BULLET = re.compile(r"^\s*[-*•]\s+", re.M)
_NUMBERED = re.compile(r"^\s*(?:\*\*)?\d+[.)]\s", re.M)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
_DIVIDER = re.compile(r"^\s*---\s*$", re.M)
_PAREN_LATIN = re.compile(r"\(([A-Za-z][A-Za-z\- ]{1,40})\)")
_CURRENCY = re.compile(r"(?<![\\$])\$\d[\d,]*(?:\.\d+)?(?=[\s,.;:!?)]|$)")
_BAD_DELIMS = re.compile(r"\\\(|\\\[")
_MENTIONS = {
    "en": re.compile(r"\b(?:grade|class|std\.?|standard)\s*\d{1,2}\b|\b\d{1,2}(?:st|nd|rd|th)[- ]grade\b|"
                     r"\b(?:CBSE|ICSE|ISC|NCERT|IGCSE|Samacheer|International Baccalaureate)\b|\bIB\b|"
                     r"\bstate board\b|\b\d{1,2}[- ]years?[- ]old\b|\byour (?:grade|class|board|age)\b", re.I),
    "hi": re.compile(r"कक्षा\s*\d|\d+\s*(?:वीं|वी)\s*कक्षा|CBSE|ICSE|NCERT"),
    "ta": re.compile(r"\d+\s*(?:ஆம்|ம்)?\s*வகுப்பு|வகுப்பு\s*\d|CBSE|ICSE|NCERT|சமச்சீர்"),
    "ar": re.compile(r"الصف\s*\S{0,10}\s*\d|\d+\s*الصف|CBSE|ICSE|IGCSE"),
}
_SCI_MATH = re.compile(r"math|science|physics|chemistry|biology", re.I)


def is_emoji(ch: str) -> bool:
    o = ord(ch)
    return o >= 0x1F000 or 0x2600 <= o <= 0x27BF or 0x2B00 <= o <= 0x2BFF


def sentences(text: str) -> list[str]:
    prose = strip_math_and_code(text)
    prose = _TABLE_ROW.sub(" ", prose)
    return [s for s in _SENT_SPLIT.split(prose) if re.search(r"\w", s)]


def sentence_lengths(text: str) -> list[int]:
    return [len([t for t in re.findall(r"\S+", s) if re.search(r"\w", t)]) for s in sentences(text)]


def script_purity(text: str, lang: str, allowed_terms: list[str]) -> Optional[float]:
    prose = strip_math_and_code(text)
    for t in allowed_terms:
        prose = prose.replace(t, " ")
    target = SCRIPT_PREFIX.get(lang)
    good = other = 0
    for ch in prose:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if name.startswith(target):
            good += 1
        else:
            other += 1
    return round(good / (good + other), 3) if good + other else None


def parenthetical_allowed(c: Case) -> bool:
    return (c.lang in ("hi", "ta", "ar") and c.sm and (c.grade or 0) >= 9
            and (c.subject is None or bool(_SCI_MATH.search(c.subject))))


def numbers_correct(text: str, answers: list) -> Optional[bool]:
    if not answers:
        return None
    tail = text[int(len(text) * 0.4):]
    ok = True
    for a in answers:
        n = str(int(a)) if float(a).is_integer() else str(a)
        pat = re.compile(rf"(?<![\d.]){re.escape(n)}(?:\.0+)?(?![\d.])")
        if not (pat.search(tail) or any(pat.search(l) for l in text.splitlines() if "=" in l or "answer" in l.lower())):
            ok = False
    if len(answers) == 2:   # quadratic: both roots stated as solutions
        roots = re.findall(r"x\s*=\s*(-?\d+)", text)
        ok = ok and {str(int(a)) for a in answers} <= set(roots)
    return ok


def truncation_suspect(text: str) -> Optional[str]:
    t = text.rstrip()
    if not t:
        return "empty"
    if t.count("```") % 2:
        return "unclosed code fence"
    if (len(re.findall(r"(?<!\\)\$\$", t)) % 2) or (len(re.findall(r"(?<![\\$])\$(?!\$)", t)) % 2):
        return "unbalanced $"
    if not re.search(r"[.!?।؟)\]*_`$:\"'”’|>]\s*$|[஀-௿ऀ-ॿ؀-ۿ]\s*$", t):
        return "no terminal punctuation"
    return None


async def auto_checks(c: Case, text: str) -> dict:
    band = grade_band(c.grade) if c.sm else None
    out: dict = {}
    words = word_count(text)
    out["words"] = words
    if c.qkind == "greeting":
        # One-word interjections ("Hi!", "வணக்கம்!") are not counted as sentences.
        sc = sum(1 for n in sentence_lengths(text) if n >= 2)
        out["length_ok"] = sc <= GREETING_MAX_SENTENCES
        out["length_detail"] = f"{sc} sentences of ≥2 words (≤{GREETING_MAX_SENTENCES})"
    elif c.qkind == "off_subject":
        _, hi = target_words(band, "summary", c.lang)
        out["length_ok"] = words <= hi * (1 + LENGTH_TOLERANCE)
        out["length_detail"] = f"{words} words (brief: ≤{hi}+25%)"
    else:
        lo, hi = target_words(band, c.length, c.lang)
        extra = 200 if c.three_ways else 0
        out["length_ok"] = lo * (1 - LENGTH_TOLERANCE) <= words <= (hi + extra) * (1 + LENGTH_TOLERANCE)
        out["length_detail"] = f"{words} words vs {lo}–{hi}{'+3ways' if extra else ''} (±25%)"
    sl = sentence_lengths(text)
    out["avg_sentence_words"] = round(statistics.mean(sl), 1) if sl else 0
    out["max_sentence_words"] = max(sl) if sl else 0
    if band and c.lang == "en" and sl:
        lim = SENTENCE_LIMIT[band]
        out["sentence_ok"] = out["avg_sentence_words"] <= lim * 1.25 and out["max_sentence_words"] <= lim * 2.5
    out["structure"] = {"headings": len(_HEADING.findall(text)), "bullets": len(_BULLET.findall(text)),
                        "numbered": len(_NUMBERED.findall(text)), "table_rows": len(_TABLE_ROW.findall(text)),
                        "divider": len(_DIVIDER.findall(text))}
    labels = sum(lbl in text for lbl in ("Analogy", "Technical", "Real-World Example"))
    if c.lang == "en" and c.qkind in ("concept", "problem"):
        out["three_ways_ok"] = (labels == 3) if c.three_ways else (labels < 2)
    if c.lang == "en" and c.qkind in ("concept", "problem"):
        exam_like = bool(re.search(r"exam (?:notes?|tips?|corner)|\b\d[- ]mark|mark-scheme|command term", text, re.I))
        if (c.grade or 0) < 9 or not c.sm:
            out["exam_notes_ok"] = not exam_like
    last_para = [p for p in text.strip().split("\n\n") if p.strip()][-1] if text.strip() else ""
    if c.sm and c.length != "small" and c.qkind in ("concept", "problem"):
        q = "?" in last_para or "؟" in last_para
        if c.personality in ("mentor", "friend"):
            out["ending_ok"] = q
        elif c.personality == "coach":
            out["ending_ok"] = q or bool(re.search(r"\b(try|challenge)\b", last_para, re.I))
        else:
            out["ending_ok"] = not last_para.rstrip().endswith("?")
    parens = [m.group(1) for m in _PAREN_LATIN.finditer(strip_math_and_code(text))] if c.lang != "en" else []
    allowed = parenthetical_allowed(c)
    if c.lang != "en":
        dup = [t for t, n in Counter(p.lower() for p in parens).items() if n > 1]
        out["parenthetical_ok"] = (not parens) if not allowed else not dup
        out["parenthetical_detail"] = f"{len(parens)} English-in-brackets {parens[:4]}{' dup:' + str(dup) if dup else ''}"
    purity = script_purity(text, c.lang, parens if allowed else [])
    out["script_purity"] = purity
    out["purity_ok"] = None if purity is None else purity >= 0.95
    m = _MENTIONS[c.lang].search(text)
    out["no_grade_board_mention"] = m is None
    if m:
        out["mention"] = m.group(0)
    segs = math_segments(text)
    out["math_segments"] = len(segs)
    if segs:
        errs = await katex_check.check_segments(segs)
        out["katex_ok"] = None if errs is None else not errs
        if errs:
            out["katex_errors"] = list(errs.values())[:2]
    out["delims_ok"] = not _BAD_DELIMS.search(text)
    out["currency_ok"] = not _CURRENCY.search(text)
    out["no_emoji"] = not any(is_emoji(ch) for ch in text)
    out["truncation"] = truncation_suspect(text)
    if c.qkind == "problem":
        out["correct"] = numbers_correct(text, c.answers or [])
    if c.qkind == "concept" and c.sm and c.grade and c.grade <= 3:
        out["no_formula_young"] = not any(re.search(r"=|\\ce|\\frac|->", s) for s in segs)
    if c.qkind == "rag":
        low = text.lower()
        out["rag_markers"] = {k: [m_ for m_ in v if m_ in low] for k, v in RAG_MARKERS.items()}
        out["rag_both_sources"] = all(out["rag_markers"].values())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Judge
# ─────────────────────────────────────────────────────────────────────────────

class Judge:
    def __init__(self, provider: str, model: str):
        self.provider, self.model = provider, model
        self.usage = {"calls": 0, "input": 0, "output": 0}
        if provider == "claude":
            import anthropic
            self.client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        else:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def json(self, system: str, user: str, max_tokens: int = 700) -> dict:
        if self.provider == "claude":
            r = await self.client.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                                  messages=[{"role": "user", "content": user}])
            text, tin, tout = r.content[0].text, r.usage.input_tokens, r.usage.output_tokens
        else:
            r = await self.client.chat.completions.create(
                model=self.model, max_tokens=max_tokens, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            text, tin, tout = r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens
        self.usage["calls"] += 1
        self.usage["input"] += tin
        self.usage["output"] += tout
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {"_unparsed": (text or "")[:300]}


RUBRIC_DIMS = ["grade_fit", "board_alignment", "difficulty_fit", "length_fit", "personality_fit",
               "language_quality", "format_compliance", "factual_correctness"]
RUBRIC_SYSTEM = ("You are a strict, experienced school teacher and curriculum reviewer. Grade the AI tutor's answer "
                 "against the exact settings it was given. For each dimension give an integer score 1-5 (5 = fully "
                 "right for the settings) and a one-line reason; use null when a dimension does not apply. "
                 "Reply with ONLY a JSON object.")


def rubric_prompt(c: Case, answer: str, scope: str) -> str:
    learner = (f"Student mode ON — Grade {c.grade}, board {c.board or 'none'}, subject {c.subject or 'not set'}."
               if c.sm else "Student mode OFF (general user): grade/board must not shape the answer.")
    fmt = f"The user explicitly asked for a format: {c.fmt}." if c.fmt else "No specific format requested."
    return f"""SETTINGS
{learner}
Language: {LANG_NAME[c.lang]} · Length: {c.length} · Difficulty: {c.difficulty} ({'relative to the grade' if c.sm else 'absolute'})
Personality: {c.personality} · Explain in 3 Ways: {'on' if c.three_ways else 'off'} · Examples: {'on' if c.examples else 'off'}
{fmt}
Rules: never mention grade/board/age; math in $…$ LaTeX; no emojis; greetings get 1–2 sentences.

{scope}

QUESTION: {c.question}

ANSWER:
<<<
{answer}
>>>

Return JSON with keys {RUBRIC_DIMS}, each {{"score": int|null, "reason": "<=20 words"}}."""


BLIND_SYSTEM = ("You are an expert curriculum reviewer. You will see several answers to the same question. The "
                "settings used to produce them are hidden. Reply with ONLY a JSON object.")


async def blind_assign(judge: Judge, question: str, answers: dict[str, str], labels: list[str], what: str,
                       hint: str) -> dict:
    """answers: true_label → text. Returns {letter: guessed_label} and the truth."""
    items = list(answers.items())
    random.shuffle(items)
    letters = [chr(65 + i) for i in range(len(items))]
    body = "\n\n".join(f"### Answer {L}\n{t[:3500]}" for L, (_, t) in zip(letters, items))
    prompt = (f"Question asked: {question}\n\nEach answer below was written for a different {what}; every one of "
              f"these {what} values is used exactly once: {labels}.\n{hint}\n\n{body}\n\n"
              f"Return JSON {{\"assignment\": {{\"A\": \"<{what}>\", ...}}}} using exactly the values listed.")
    res = await judge.json(BLIND_SYSTEM, prompt, max_tokens=400)
    guess = (res.get("assignment") or {}) if isinstance(res, dict) else {}
    return {"truth": {L: lbl for L, (lbl, _) in zip(letters, items)},
            "guess": {L: str(guess.get(L)) for L in letters}}


async def blind_order(judge: Judge, question: str, answers: dict[str, str]) -> dict:
    items = list(answers.items())
    random.shuffle(items)
    letters = [chr(65 + i) for i in range(len(items))]
    body = "\n\n".join(f"### Answer {L}\n{t[:3500]}" for L, (_, t) in zip(letters, items))
    prompt = (f"Question asked: {question}\n\nThese answers were written for the SAME learner at different difficulty "
              f"settings. Order them from EASIEST to HARDEST (conceptual depth, rigour, assumed knowledge).\n\n{body}\n\n"
              "Return JSON {\"easiest_to_hardest\": [\"<letter>\", ...]}.")
    res = await judge.json(BLIND_SYSTEM, prompt, max_tokens=200)
    order = res.get("easiest_to_hardest") or []
    truth = {L: lbl for L, (lbl, _) in zip(letters, items)}
    return {"truth": truth, "order": [truth.get(L) for L in order]}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def spearman(x: list[float], y: list[float]) -> Optional[float]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return round(float(np.corrcoef(rx, ry)[0, 1]), 3)


def vocab_score(text: str) -> float:
    toks = [t for t in re.findall(r"[A-Za-z]+", strip_math_and_code(text))]
    return round(sum(len(t) >= 7 for t in toks) / len(toks), 3) if toks else 0.0


def cosine(a: str, b: str) -> float:
    ca = Counter(re.findall(r"\w+", strip_math_and_code(a).lower()))
    cb = Counter(re.findall(r"\w+", strip_math_and_code(b).lower()))
    num = sum(ca[k] * cb[k] for k in ca.keys() & cb.keys())
    den = (sum(v * v for v in ca.values()) ** 0.5) * (sum(v * v for v in cb.values()) ** 0.5)
    return round(num / den, 3) if den else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Cost estimate
# ─────────────────────────────────────────────────────────────────────────────

_OUT_TOKENS = {"small": 120, "summary": 350, "detailed": 800, "deep-dive": 1600}
_LANG_TOK = {"en": 1.0, "hi": 2.2, "ta": 2.6, "ar": 2.0}


def price(model: str, tin: int, tout: int) -> Optional[float]:
    p = MODEL_PRICES_USD_PER_MTOK.get(model)
    return None if p is None else (tin * p[0] + tout * p[1]) / 1e6


def estimate(cases: list[Case], gen_model: str, judge_model: str, judge_on: bool) -> dict:
    gens = sum(len(c.midchat) if c.midchat else 1 for c in cases)
    out_tok = sum(int(_OUT_TOKENS[c.length] * _LANG_TOK[c.lang]) if c.qkind != "greeting" else 40 for c in cases)
    in_tok = gens * 2600
    enh = sum(4 for c in cases if c.enhancements)
    rubric = gens if judge_on else 0
    blind = (12 if judge_on else 0)
    j_in = rubric * 3200 + blind * 9000
    j_out = rubric * 450 + blind * 250
    return {"generations": gens, "enhancement_calls": enh, "judge_rubric_calls": rubric, "judge_blind_calls": blind,
            "gen_tokens_est": (in_tok, out_tok), "judge_tokens_est": (j_in, j_out),
            "gen_cost": price(gen_model, in_tok + enh * 1500, out_tok + enh * 400),
            "judge_cost": price(judge_model, j_in, j_out) if judge_on else 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

async def run_cases(be: LocalBackend, cases: list[Case]) -> list[dict]:
    records = []
    vault_id = None
    if any(c.qkind == "rag" for c in cases):
        vault_id = await be.upload_vault(VAULT_DOC)
    for i, c in enumerate(cases, 1):
        rec = {"case": asdict(c)}
        try:
            await be.put_context(c)
            chat_id, cs = await be.new_chat(c)
            if c.midchat:
                turns = []
                for q, tw in c.midchat:
                    if tw != cs.get("explain_3ways"):
                        await be.patch_settings(chat_id, {"explain_3ways": tw})
                        cs = await be.get_settings(chat_id)
                    r = await be.stream(chat_id, q, be.chat_context(c, cs), cs)
                    turns.append({"question": q, "three_ways": tw, **r})
                rec["turns"] = turns
                last = turns[-1]["answer"]
                rec["result"] = {**turns[-1]}
                rec["checks"] = {"labels_turn1": sum(l in turns[0]["answer"] for l in ("Analogy", "Technical", "Real-World Example")),
                                 "labels_last": sum(l in last for l in ("Analogy", "Technical", "Real-World Example"))}
                rec["checks"]["three_ways_stops_ok"] = rec["checks"]["labels_last"] < 2
            else:
                extra = {}
                if c.qkind == "rag":
                    extra = {"inline_docs": [ATTACHED_DOC], "selected_files": [vault_id] if vault_id else []}
                r = await be.stream(chat_id, c.question, be.chat_context(c, cs, extra), cs)
                rec["result"] = r
                rec["checks"] = await auto_checks(c, r["answer"])
                if c.qkind == "rag" and not vault_id:
                    rec["checks"]["rag_note"] = "vault upload/processing failed — only the attached file was sent"
                if c.enhancements and r["answer"]:
                    rec["enhancements"] = await run_enhancements(be, chat_id, c, r["answer"])
            print(f"[{i}/{len(cases)}] {c.case_id:<44} {served_label(rec)} "
                  f"words={rec['checks'].get('words', '-')} {'FALLBACK ' if rec['result'].get('fallback') else ''}",
                  flush=True)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{i}/{len(cases)}] {c.case_id} ERROR {rec['error'][:160]}", flush=True)
        records.append(rec)
    return records


async def run_enhancements(be: LocalBackend, chat_id: str, c: Case, answer: str) -> dict:
    """Same request bodies the current frontend sends (AskDocsPage.fetchEnhancements)."""
    base = {"message": c.question, "response": answer}
    calls = {
        "follow-ups": {**base, "count": 4, "language": c.lang},
        "next-steps": {**base, "count": 4, "language": c.lang},
        "practice-exercises": {**base, "count": 3, "difficulty": c.difficulty, "language": c.lang},
        "mindmap": {**base, "grade": c.grade if c.sm else None, "language": c.lang},
    }
    out = {}
    for path, body in calls.items():
        r = await be.enhancement(chat_id, path, body)
        payload = r.get("json") or {}
        items = payload.get("questions") or payload.get("steps") or payload.get("exercises") or payload.get("mindmap")
        flat = json.dumps(items, ensure_ascii=False) if items is not None else ""
        out[path] = {"status": r["status"], "seconds": r["seconds"], "count": len(items) if isinstance(items, list) else None,
                     "script_purity": script_purity(flat, c.lang, []) if flat else None, "sample": flat[:400]}
    return out


async def judge_all(judge: Judge, records: list[dict], scope: str) -> None:
    for rec in records:
        if rec.get("error") or "result" not in rec or not rec["result"].get("answer"):
            continue
        c = Case(**rec["case"])
        try:
            rec["judge"] = await judge.json(RUBRIC_SYSTEM, rubric_prompt(c, rec["result"]["answer"], scope))
        except Exception as exc:
            rec["judge"] = {"_error": f"{type(exc).__name__}: {str(exc)[:200]}"}


async def blind_tests(judge: Judge, records: list[dict]) -> dict:
    by = defaultdict(dict)
    for r in records:
        if r.get("error") or r["case"]["qkind"] == "midchat":
            continue
        by[(r["case"]["sweep"], r["case"]["group"])][r["case"]["factor"]] = r
    out = {"grade": [], "board": [], "personality": [], "length": [], "difficulty": []}
    for (sweep, grp), fac in by.items():
        q = next(iter(fac.values()))["case"]["question"]
        if sweep == "S1":
            sel = {f"Grade {r['case']['grade']}": r["result"]["answer"] for f, r in fac.items() if r["case"]["grade"] in (2, 5, 8, 11)}
            if len(sel) == 4:
                out["grade"].append(await blind_assign(judge, q, sel, sorted(sel), "grade",
                                                       "Grades are school years (Grade 2 ≈ age 7, Grade 11 ≈ age 16)."))
        elif sweep == "S2":
            for g in {r["case"]["grade"] for r in fac.values()}:
                sel = {r["case"]["board"]: r["result"]["answer"] for r in fac.values()
                       if r["case"]["grade"] == g and r["case"]["board"] in BOARDS_BLIND}
                if len(sel) == 5:
                    out["board"].append(await blind_assign(judge, q, sel, BOARDS_BLIND, "curriculum board",
                                                           "Boards: CBSE (NCERT), ICSE, a generic Indian State board, IB, Cambridge."))
        elif sweep == "S5":
            for g in {r["case"]["grade"] for r in fac.values()}:
                sel = {r["case"]["personality"]: r["result"]["answer"] for r in fac.values() if r["case"]["grade"] == g}
                if len(sel) == 5:
                    out["personality"].append(await blind_assign(judge, q, sel, PERSONALITIES, "tutor personality",
                                                                 "mentor=warm encouraging; coach=direct, hint-first, challenge; professor=formal rigorous; friend=casual humour; technical-expert=precise mechanisms/edge cases."))
        elif sweep == "S3":
            for g in {r["case"]["grade"] for r in fac.values()}:
                sel = {r["case"]["length"]: r["result"]["answer"] for r in fac.values() if r["case"]["grade"] == g}
                if len(sel) == 4:
                    out["length"].append(await blind_assign(judge, q, sel, LENGTHS, "length setting",
                                                            "small < summary < detailed < deep-dive."))
        elif sweep == "S4":
            for g in {r["case"]["grade"] for r in fac.values() if r["case"]["sm"]}:
                sel = {r["case"]["difficulty"]: r["result"]["answer"] for r in fac.values()
                       if r["case"]["grade"] == g and r["case"]["sm"]}
                if len(sel) == 4:
                    out["difficulty"].append(await blind_order(judge, q, sel))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

CHECK_KEYS = ["length_ok", "sentence_ok", "three_ways_ok", "exam_notes_ok", "ending_ok", "parenthetical_ok",
              "purity_ok", "no_grade_board_mention", "katex_ok", "delims_ok", "currency_ok", "no_emoji",
              "correct", "no_formula_young", "rag_both_sources", "three_ways_stops_ok"]


def _pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{sum(xs)}/{len(xs)}" if xs else "–"


def served_label(r: dict) -> str:
    res = r.get("result") or {}
    sb = res.get("served_by")
    if sb:
        return f"{sb['provider']}:{sb['model']}"
    return "canned reply (no LLM)" if res.get("answer") else "none (error/empty)"


def write_report(out: Path, records: list[dict], blind: dict, meta: dict) -> None:
    L = [f"# AI Assistant tailoring verification — {meta['mode']}", ""]
    if meta.get("label"):
        L += [f"> **{meta['label']}**", ""]
    L += [f"Run {meta['run_at']} · backend {meta['base_url']} (git {meta.get('backend_sha')}) · user {meta['user']} · "
          f"judge {meta['judge'] or 'off'} · cases {len(records)}", ""]
    mix = Counter(served_label(r) for r in records if r.get("result"))
    L += ["## Provider / fallback usage", "", "| Served by | Answers |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in mix.most_common()]
    L += ["", f"Fallbacks: {sum(1 for r in records if (r.get('result') or {}).get('fallback'))} · breaker trips: "
          f"{sum((r.get('result') or {}).get('breaker_trips', 0) for r in records)} · guard fail-open: "
          f"{sum((r.get('result') or {}).get('guard_fail_open', 0) for r in records)} · interrupted (G6): "
          f"{sum(1 for r in records if (r.get('result') or {}).get('interrupted'))} · errors: "
          f"{sum(1 for r in records if r.get('error'))}", ""]

    # per sweep × check
    sweeps = sorted({r["case"]["sweep"] for r in records}, key=lambda s: int(s[1:]))
    L += ["## Automatic checks per sweep (passed / applicable)", "",
          "| Sweep | " + " | ".join(CHECK_KEYS) + " |", "|" + "---|" * (len(CHECK_KEYS) + 1)]
    for s in sweeps:
        rs = [r for r in records if r["case"]["sweep"] == s and r.get("checks")]
        L.append(f"| {s} | " + " | ".join(_pct([r["checks"].get(k) for r in rs]) for k in CHECK_KEYS) + " |")
    tr = [(r["case"]["case_id"], r["checks"]["truncation"]) for r in records if (r.get("checks") or {}).get("truncation")]
    L += ["", f"Truncation suspects (heuristic — finish_reason is not visible over HTTP): {tr or 'none'}", ""]

    # judge
    judged = [r for r in records if isinstance(r.get("judge"), dict) and not r["judge"].get("_error")]
    if judged:
        L += ["## Judge rubric (1–5, settings visible)", "", "| Sweep | " + " | ".join(RUBRIC_DIMS) + " |",
              "|" + "---|" * (len(RUBRIC_DIMS) + 1)]
        for s in sweeps:
            rs = [r for r in judged if r["case"]["sweep"] == s]
            cells = []
            for d in RUBRIC_DIMS:
                v = [((r["judge"].get(d) or {}).get("score")) for r in rs]
                v = [x for x in v if isinstance(x, (int, float))]
                cells.append(f"{statistics.mean(v):.2f}" if v else "–")
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L.append("")

    # trends S1
    s1 = [r for r in records if r["case"]["sweep"] == "S1" and r.get("checks") and r["case"]["qkind"] == "concept"]
    if s1:
        by_g = defaultdict(list)
        for r in s1:
            by_g[r["case"]["grade"]].append(r)
        L += ["## Grade trends (S1, concept questions)", "", "| Grade | Avg sentence words | Words | Vocab (≥7-letter share) |",
              "|---|---|---|---|"]
        gs, sw, wc, vb = [], [], [], []
        for g in sorted(by_g):
            a = statistics.mean(r["checks"]["avg_sentence_words"] for r in by_g[g])
            w = statistics.mean(r["checks"]["words"] for r in by_g[g])
            v = statistics.mean(vocab_score(r["result"]["answer"]) for r in by_g[g])
            gs.append(g); sw.append(a); wc.append(w); vb.append(v)
            L.append(f"| {g} | {a:.1f} | {w:.0f} | {v:.3f} |")
        L += ["", f"Spearman vs grade — sentence length {spearman(gs, sw)}, words {spearman(gs, wc)}, vocabulary "
              f"{spearman(gs, vb)} (target ≥ 0.8; few points in --quick).", ""]

    # S3 length monotonicity
    s3 = [r for r in records if r["case"]["sweep"] == "S3" and r.get("checks")]
    if s3:
        L += ["## Length vs word count (S3)", "", "| Grade | Question | small | summary | detailed | deep-dive | strictly rising |",
              "|---|---|---|---|---|---|---|"]
        grp = defaultdict(dict)
        for r in s3:
            grp[(r["case"]["grade"], r["case"]["group"])][r["case"]["length"]] = r["checks"]["words"]
        for (g, gq), d in sorted(grp.items()):
            seq = [d.get(x) for x in LENGTHS]
            ok = all(a is not None and b is not None and a < b for a, b in zip(seq, seq[1:]))
            L.append(f"| {g} | {gq.split('|')[1]} | " + " | ".join(str(x) for x in seq) + f" | {'PASS' if ok else 'FAIL'} |")
        L.append("")

    # distinctness
    L += ["## Distinctness (cosine similarity of word counts; < 0.90 = clearly different)", ""]
    pairs = []
    for sweep, key in (("S1", "grade"), ("S2", "board"), ("S5", "personality")):
        grp = defaultdict(dict)
        for r in records:
            if r["case"]["sweep"] == sweep and r.get("result", {}).get("answer"):
                grp[r["case"]["group"]][str(r["case"][key])] = r["result"]["answer"]
        for g, d in grp.items():
            ks = sorted(d)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    if sweep == "S1" and abs(int(ks[i]) - int(ks[j])) != 3:
                        continue
                    pairs.append((sweep, g.split("|")[1], ks[i], ks[j], cosine(d[ks[i]], d[ks[j]])))
    if pairs:
        L += ["| Sweep | Q | A | B | similarity |", "|---|---|---|---|---|"]
        L += [f"| {s} | {q} | {a} | {b} | {sim}{' ⚠' if sim >= 0.9 else ''} |" for s, q, a, b, sim in sorted(pairs, key=lambda p: -p[4])[:25]]
        L += ["", f"Max similarity {max(p[4] for p in pairs)} · pairs ≥ 0.90: {sum(p[4] >= 0.9 for p in pairs)} of {len(pairs)}", ""]

    # blind
    if any(blind.values()):
        L += ["## Blind identification (judge, settings hidden, order shuffled)", ""]
        for kind, runs in blind.items():
            if not runs:
                continue
            if kind == "difficulty":
                rhos = []
                for b in runs:
                    order = [x for x in b["order"] if x in DIFFICULTIES]
                    if len(order) == 4:
                        rhos.append(spearman([DIFFICULTIES.index(x) for x in order], list(range(4))))
                L += [f"**Difficulty ordering** — Spearman of judged order vs set order per group: {rhos} "
                      f"(1.0 = perfectly rising easy<medium<hard<expert)", ""]
                continue
            tot = cor = 0
            conf = defaultdict(Counter)
            for b in runs:
                for letter, truth in b["truth"].items():
                    g = b["guess"].get(letter)
                    conf[truth][g] += 1
                    tot += 1
                    cor += (g == truth)
            labels = sorted(conf)
            L += [f"**{kind}** — accuracy {cor}/{tot} = {cor / tot:.0%} (chance {1 / len(labels):.0%})", "",
                  "| truth \\ guess | " + " | ".join(labels) + " | other |", "|" + "---|" * (len(labels) + 2)]
            for t in labels:
                L.append(f"| {t} | " + " | ".join(str(conf[t][g]) for g in labels) + f" | {sum(v for g, v in conf[t].items() if g not in labels)} |")
            L.append("")

    # correctness
    corr = [r for r in records if (r.get("checks") or {}).get("correct") is not None]
    if corr:
        L += ["## Correctness (problem-solving questions)", "",
              f"{sum(r['checks']['correct'] for r in corr)}/{len(corr)} answers state the stored numeric answer.", ""]
        bad = [r for r in corr if not r["checks"]["correct"]]
        L += [f"- {r['case']['case_id']} ({r['case']['lang']}): expected {r['case']['answers']}" for r in bad[:10]]
        L.append("")

    # enhancements
    enh = [r for r in records if r.get("enhancements")]
    if enh:
        L += ["## Enhancements (subset; same bodies as the frontend)", "", "| Case | Endpoint | HTTP | Items | Script purity | Sample |",
              "|---|---|---|---|---|---|"]
        for r in enh:
            for path, e in r["enhancements"].items():
                L.append(f"| {r['case']['case_id']} | {path} | {e['status']} | {e['count']} | {e['script_purity']} | "
                         f"{(e['sample'] or '')[:90].replace('|', '/')} |")
        L.append("")

    # weakest 10
    def score(r):
        c = r.get("checks") or {}
        fails = sum(1 for k in CHECK_KEYS if c.get(k) is False)
        j = r.get("judge") or {}
        js = [((j.get(d) or {}).get("score")) for d in RUBRIC_DIMS]
        js = [x for x in js if isinstance(x, (int, float))]
        return (-(fails), statistics.mean(js) if js else 5)
    weakest = sorted([r for r in records if r.get("result")], key=score)[:10]
    L += ["## 10 weakest cases", ""]
    for r in weakest:
        c, ch = r["case"], r.get("checks") or {}
        failed = [k for k in CHECK_KEYS if ch.get(k) is False]
        L += [f"### {c['case_id']} — failed: {failed or 'none'}",
              f"Settings: grade {c['grade']}, {c['board']}, {c['lang']}, sm={c['sm']}, {c['length']}, {c['difficulty']}, "
              f"{c['personality']}, 3ways={c['three_ways']}, examples={c['examples']}, fmt={c['fmt']}",
              f"Likely fragment: {likely_fragment(failed, c)}",
              f"Judge: {json.dumps({d: (r.get('judge') or {}).get(d, {}).get('reason') for d in RUBRIC_DIMS if isinstance((r.get('judge') or {}).get(d), dict) and ((r.get('judge') or {}).get(d) or {}).get('score', 5) and ((r.get('judge') or {}).get(d) or {}).get('score', 5) <= 3}, ensure_ascii=False)[:400]}",
              "", "> " + r["result"]["answer"][:500].replace("\n", "\n> "), ""]

    # cost
    L += ["## Tokens & cost", "", f"- HTTP generations: {meta['generations']} (token usage is not exposed over HTTP; "
          f"estimated output ≈ {meta['gen_out_est']:,} tokens from answer length).",
          f"- Judge ({meta['judge'] or 'off'}): {meta['judge_usage']['calls']} calls, {meta['judge_usage']['input']:,} in / "
          f"{meta['judge_usage']['output']:,} out → {meta['judge_cost']}.",
          f"- Generation cost estimate: {meta['gen_cost']}.", ""]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")


def likely_fragment(failed: list[str], c: dict) -> str:
    m = {"length_ok": "response_settings Length line (settings.length_line / matrices.LENGTH_WORDS)",
         "sentence_ok": "learner_profile band fragment (learner.band_fragment)",
         "three_ways_ok": "response_settings three_ways_line / THREE_WAYS_OFF_AFTER_USE",
         "exam_notes_ok": "learner.wants_exam_notes / _EXAM_NOTES",
         "ending_ok": "settings.personality_line (_ENDING)",
         "parenthetical_ok": "language.language_section (D9 parenthetical rule)",
         "purity_ok": "language.language_section + language_reminder",
         "no_grade_board_mention": "hard rule 2 (base.HARD_RULES)",
         "katex_ok": "hard rule 4 (math)", "delims_ok": "hard rule 4 (math)", "currency_ok": "hard rule 4 (currency)",
         "no_emoji": "base.ASSISTANT_ONLY_RULES", "correct": "model accuracy (no prompt fragment)",
         "no_formula_young": "learner.band_fragment early-primary scope",
         "rag_both_sources": "router RAG injection (G2, Phase 4)", "three_ways_stops_ok": "THREE_WAYS_OFF_AFTER_USE"}
    return "; ".join(m.get(f, f) for f in failed) or "— (judge-only weakness)"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

async def main_async(a) -> None:
    bank = load_bank()
    cases = build_cases(bank, a.quick, a.samples or (1 if a.quick else 2))
    if a.sweeps:
        keep = set(a.sweeps.split(","))
        cases = [c for c in cases if c.sweep in keep]
    judge_on = a.judge_model.lower() != "off"
    est = estimate(cases, a.assume_generator_model, a.judge_model, judge_on)
    fmt = lambda v: "price unknown" if v is None else f"${v:.2f}"
    print(f"Cases: {len(cases)} · HTTP generations: {est['generations']} · enhancement calls: {est['enhancement_calls']} · "
          f"judge rubric calls: {est['judge_rubric_calls']} · judge blind calls (≈): {est['judge_blind_calls']}")
    print(f"Estimated tokens — generation {est['gen_tokens_est'][0]:,} in / {est['gen_tokens_est'][1]:,} out "
          f"({a.assume_generator_model}: {fmt(est['gen_cost'])}); judge {est['judge_tokens_est'][0]:,} in / "
          f"{est['judge_tokens_est'][1]:,} out ({a.judge_model}: {fmt(est['judge_cost']) if judge_on else 'off'})")
    print("Per sweep:", dict(Counter(c.sweep for c in cases)))
    if a.dry_run:
        return

    out = Path(a.out or f"eval_results/tailoring_{date.today():%Y%m%d}{'_quick' if a.quick else ''}")
    out.mkdir(parents=True, exist_ok=True)
    be = LocalBackend(a.base_url, a.email, a.password, Path(a.backend_log) if a.backend_log else None)
    await be.login()
    health = (await be.client.get("/health")).json()
    records = await run_cases(be, cases)
    judge = blind = None
    blind = {}
    if judge_on:
        judge = Judge(a.judge_provider, a.judge_model)
        await judge_all(judge, records, bank["scope_table"])
        blind = await blind_tests(judge, records)
    gen_out_est = sum(int(len((r.get("result") or {}).get("answer", "")) / 3.5) for r in records)
    usage = judge.usage if judge else {"calls": 0, "input": 0, "output": 0}
    jc = price(a.judge_model, usage["input"], usage["output"]) if judge else 0.0
    meta = {"mode": "quick" if a.quick else "full", "label": a.label, "run_at": time.strftime("%Y-%m-%d %H:%M"),
            "base_url": a.base_url, "backend_sha": health.get("git_sha_short"), "user": a.email,
            "judge": f"{a.judge_provider}:{a.judge_model}" if judge_on else None, "generations": est["generations"],
            "gen_out_est": gen_out_est, "judge_usage": usage, "judge_cost": fmt(jc),
            "gen_cost": "see provider mix; " + fmt(price(a.assume_generator_model, est['gen_tokens_est'][0], gen_out_est))
                        + f" if all served by {a.assume_generator_model}"}
    with open(out / "raw.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "blind.json").write_text(json.dumps(blind, ensure_ascii=False, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    write_report(out, records, blind, meta)
    print(f"\nREPORT → {out / 'REPORT.md'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--samples", type=int)
    p.add_argument("--sweeps", help="comma list, e.g. S1,S3")
    p.add_argument("--dry-run", action="store_true", help="print case counts and cost estimate only")
    p.add_argument("--base-url", default="http://127.0.0.1:8001")
    p.add_argument("--backend-log", help="local backend log file (INFO logging for 'app') — provider attribution")
    p.add_argument("--email", default="tailoring-eval@genverse.dev")
    p.add_argument("--password", default="TailorEval#2026")
    p.add_argument("--judge-provider", choices=["claude", "openai"], default="claude")
    p.add_argument("--judge-model", default="claude-sonnet-5-5", help="'off' to skip the judge")
    p.add_argument("--assume-generator-model", default=settings.AI_PRIMARY_MODEL,
                   help="model used for the generation cost estimate")
    p.add_argument("--label", help="banner shown at the top of REPORT.md (e.g. 'NOT production routing')")
    p.add_argument("--out")
    p.add_argument("--report-only", help="rebuild REPORT.md from an existing run directory")
    a = p.parse_args()
    if a.report_only:
        d = Path(a.report_only)
        records = [json.loads(line) for line in open(d / "raw.jsonl")]
        blind = json.loads((d / "blind.json").read_text()) if (d / "blind.json").exists() else {}
        write_report(d, records, blind, json.loads((d / "meta.json").read_text()))
        print(f"REPORT → {d / 'REPORT.md'}")
        return
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
