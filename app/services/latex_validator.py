"""Server-side LaTeX validator for AI-generated assessment questions.

Pure Python, no external runtime dependency. Three structural checks run on
every question before it's saved:

1. Delimiter balance — every $ that opens inline math must find a matching
   closing $ (and $$ / $$) in the same string. Catches the "$15.90\\%"
   (opening $ with no closing $) bug directly.
2. Balanced braces within each math segment — catches things like
   "\\frac{1}{2" (a macro argument left open).
3. Bare chemical formulas — a segment that's ENTIRELY made of
   element-symbol+digit groups (e.g. "NaHCO3", "Na2CO3") with no \\ce{...}
   wrapper. KaTeX happily parses this as ordinary math (each letter becomes
   an italic variable) so it never raises, which is exactly why the original
   bug wasn't a "parse error" — it's a semantic issue, not a syntax one, so a
   syntax-only check like real KaTeX parsing would never have caught it any
   more than this regex does.

This trades exhaustive coverage (a truly malformed \\begin{matrix}, a
mismatched \\left/\\right, etc. can still slip through) for zero external
dependencies — the backend stays 100% Python, nothing else needs installing
wherever this runs. It only flags; callers decide whether to log, drop, or
surface flagged questions.
"""
from __future__ import annotations

import re

# The ONE definition of which question types carry an "options"
# array/correct_answer that needs label-stripping, shared by every caller
# that applies these rules (app/services/ai_service.py, app/routers/
# evaluation.py, scripts/migrate_latex_normalize.py, scripts/
# migrate_evaluation_latex_normalize.py) — previously each of those
# reimplemented the same literal tuples independently, so a future new
# question type had to be added in lockstep across all of them or the
# schema-consistency rule silently broke in whichever copy was missed.
NO_OPTION_TYPES = ("fill", "short", "long")
OPTION_BEARING_TYPES = ("mcq", "true_false", "match")
CORRECT_ANSWER_STRIP_TYPES = ("mcq", "true_false")

_DISPLAY_RE = re.compile(r"\$\$([\s\S]+?)\$\$")
_INLINE_RE = re.compile(r"(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)")

# All 118 periodic-table symbols, longest first so regex alternation tries
# two-letter symbols (Na, Cl, Ca, ...) before a colliding single-letter one
# (N, C) matches its prefix instead. A generic "[A-Z][a-z]?" class is too
# loose here — real assessment content hit false positives on bare algebra
# like matrix products "AB"/"BA" ("A" isn't an element, but a looser class
# would still match it as one letter + one letter). Requiring every group to
# be an ACTUAL element symbol makes "AB" fail to match at all while still
# catching "NaHCO3", "HCl", "Na2CO3", etc.
_ELEMENT_SYMBOLS = sorted(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
    Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce
    Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
    Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl
    Mc Lv Ts Og""".split(),
    key=len,
    reverse=True,
)
# Matches a segment that is NOTHING BUT two-or-more concatenated
# element-symbol+digit groups, e.g. "NaHCO3" (Na, H, C, O3), "Na2CO3"
# (Na2, C, O3), "H2SO4" (H2, S, O4), "CO2" (C, O2). Anchored full-match so it
# never fires on a segment that also contains other math (an equals sign, an
# operator, prose) — those aren't the "bare formula wrapped in plain $...$"
# shape the bug report described.
_BARE_CHEMICAL_FORMULA_RE = re.compile(
    r"^(?:(?:" + "|".join(_ELEMENT_SYMBOLS) + r")\d*){2,}$"
)


def extract_math_segments(text: str) -> list[str]:
    """Every balanced $$...$$ and $...$ segment's inner content, in order."""
    if not text:
        return []
    segments = _DISPLAY_RE.findall(text)
    # Strip display math first so its $ characters can't be mistaken for
    # inline delimiters when scanning what's left.
    remainder = _DISPLAY_RE.sub(" ", text)
    segments += _INLINE_RE.findall(remainder)
    return segments


def has_unbalanced_dollar(text: str) -> bool:
    """True if a $ in `text` never finds a matching closing $ — the exact
    "$15.90\\%" (no closing $) shape seen in production."""
    if not text or "$" not in text:
        return False
    remainder = _INLINE_RE.sub("", _DISPLAY_RE.sub("", text))
    return "$" in remainder


def has_unbalanced_braces(segment: str) -> bool:
    """True if `{`/`}` in a math segment don't balance — e.g. "\\frac{1}{2"
    (a macro argument that was never closed). Escaped braces (\\{, \\})
    are literal glyphs in TeX, not grouping, so they're skipped."""
    depth = 0
    i = 0
    n = len(segment)
    while i < n:
        c = segment[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return True
        i += 1
    return depth != 0


def looks_like_bare_chemical_formula(segment: str) -> bool:
    """True if a math segment is nothing but a chemical formula's element
    symbols with no \\ce{...} macro wrapping it — the exact NaHCO3/Na2CO3/HCl
    shape from the bug report, which renders as bare italic math variables
    instead of upright chemistry notation."""
    stripped = segment.strip()
    if "\\" in stripped:
        return False
    return bool(_BARE_CHEMICAL_FORMULA_RE.match(stripped))


def repair_unbalanced_leading_dollar(text: str) -> str:
    """Closes a lone unbalanced leading $ (e.g. "$15.90\\%" with no closing
    $) by appending one. Mirrors the frontend's normalizeLatex repair
    (remix-of-genverse-eduverse/src/lib/latex-utils.ts) — kept conservative
    for the same reason: only fires on a short, single-line string whose
    ENTIRE content starts with one $ and has no other unescaped $ later."""
    if not text or not text.startswith("$") or text.startswith("$$"):
        return text
    if len(text) > 80 or "\n" in text:
        return text
    rest = text[1:]
    if re.search(r"(?<!\\)\$", rest):
        return text
    return text + "$"


def _collapse_stray_triple_dollar(text: str) -> str:
    """Collapses a stray extra "$" the model sometimes appends right after a
    legitimate "$$...$$" close (e.g. "...^{1/n}$$$." — confirmed real
    generation output, caught via a screenshot of a rendered question) back
    down to a clean "$$". Mirrors the frontend's identical repair
    (remix-of-genverse-eduverse/src/shared/math-content/normalizeMath.ts,
    collapseStrayTripleDollar) — see its comment for why exactly 3 "$" is
    unambiguous (never valid syntax on its own) and why the match is scoped
    to only fire before whitespace/punctuation/end-of-string, so a
    display-math block immediately followed by a genuine inline formula with
    no separator is left untouched."""
    return re.sub(r"\${3}(?=[\s.,;:!?)\]]|$)", "$$", text)


def repair_common_latex_issues(text: str) -> str:
    """Safe, conservative repairs applied before validation/save — used by
    both the one-off DB migration script (scripts/migrate_latex_normalize.py)
    and finalize_generated_questions. Mirrors the frontend normalizer's
    fixes: unbalanced leading $, double-escaped macro backslashes, a stray
    extra "$" after a display-math close, and a stray \\% that sits outside
    any math span."""
    if not text:
        return text

    result = repair_unbalanced_leading_dollar(text)
    result = _collapse_stray_triple_dollar(result)
    # \\ce{...} -> \ce{...}: an LLM sometimes double-escapes a backslash when
    # asked to copy LaTeX verbatim into a JSON string.
    result = re.sub(r"\\{2,}([a-zA-Z]+\{)", r"\\\1", result)

    protected: list[str] = []

    def _protect(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"\x02P{len(protected) - 1}\x03"

    result = _DISPLAY_RE.sub(_protect, result)
    result = _INLINE_RE.sub(_protect, result)
    # Whatever's left is outside any math span — a literal \% here is prose
    # that over-escaped a percent sign, not valid TeX.
    result = result.replace("\\%", "%")
    for i, original in enumerate(protected):
        result = result.replace(f"\x02P{i}\x03", original, 1)

    return result


_OPTION_PREFIX_RE = re.compile(r"^\s*([A-Za-z]|\d{1,2})[.)]\s+")


def _expected_option_labels(index: int) -> set[str]:
    """The label(s) a duplicate-prefix bug would plausibly use for the
    option AT THIS POSITION — "A"/"a" for index 0, "B"/"b" for index 1, ...,
    plus the 1-based digit form ("1", "2", ...). Anything else at the start
    of this option's text is real content, not a stray label."""
    labels = {str(index + 1)}
    if 0 <= index < 26:
        letter = chr(ord("A") + index)
        labels.add(letter)
        labels.add(letter.lower())
    return labels


def strip_option_prefix(text: str, index: int | None = None) -> str:
    """Strips a leading "A. "/"B) "/"1. " style label the model sometimes adds
    to an option despite being told not to (the prompt's own option letters
    are assigned by the frontend, not embedded in the stored text) — e.g.
    "A. 2" -> "2". Conservative: only strips a SINGLE short label at the very
    start, never touches anything else in the string.

    `index` is this option's 0-based position in its own options list, when
    known. A bare regex match on "letter/digit + '.'/')' + space" is NOT
    enough on its own — real option content legitimately starts that way
    (an option whose actual answer is "A. Einstein" or a match-question item
    "1. Inertia"), and blindly stripping it silently corrupts that content.
    When `index` is given, the match is only treated as a stray duplicate
    label if it's the label that would ACTUALLY be assigned to this
    position (e.g. only "A."/"1." at index 0, only "B."/"2." at index 1) —
    the one case a model erroneously prefixing its own answer key would
    actually produce. `index=None` (e.g. for a correct_answer string that
    isn't itself a positioned list entry) falls back to the older
    unconditional strip, kept for backward compatibility."""
    if not text:
        return text
    m = _OPTION_PREFIX_RE.match(text)
    if not m:
        return text
    if index is not None and m.group(1) not in _expected_option_labels(index):
        return text
    return text[m.end():]


def repair_correct_answer(raw_correct_answer, original_options, repaired_options) -> str:
    """Derives a schema-consistent correct_answer for mcq/true_false
    questions FROM the (already position-aware repaired) options list
    whenever it matches one of the original option strings, instead of
    independently re-running strip_option_prefix on it — correct_answer on
    its own carries no positional information, so a second, unguarded strip
    there would reintroduce exactly the false-positive risk `index` exists
    to prevent in strip_option_prefix (e.g. a correct_answer of "A.
    Einstein" losing "A. " even though the model never duplicated a label).
    The match is whitespace-insensitive (.strip()) so incidental spacing
    differences between an option and its answer-key copy — plausible model
    output noise, not a real content difference — don't defeat it. When
    correct_answer still doesn't match any original option (e.g. it was
    genuinely paraphrased/truncated), no positional signal exists to guard a
    strip, so only repair_common_latex_issues is applied — per the prompt's
    own instructions correct_answer is supposed to match an option verbatim,
    so a non-match at that point is already anomalous data, and guessing at
    a strip there would risk the exact corruption this function exists to
    prevent, for a case that should be rare in the first place. EITHER WAY
    the return value is already fully LaTeX-repaired — callers should use
    this in place of, not in addition to, their own repair_common_latex_issues
    call on this field."""
    text = str(raw_correct_answer) if raw_correct_answer is not None else ""
    if isinstance(original_options, list) and isinstance(repaired_options, list):
        for i, opt in enumerate(original_options):
            if option_text(opt).strip() == text.strip() and i < len(repaired_options):
                return repaired_options[i]
    return repair_common_latex_issues(text)


def option_text(opt) -> str:
    if isinstance(opt, dict):
        return str(opt.get("text", ""))
    return str(opt) if opt is not None else ""


async def validate_questions_latex(questions: list[dict]) -> dict[str, list[str]]:
    """Returns {question_id: [issue, ...]} for every question with a LaTeX
    problem in text/options/correct_answer/explanation. Empty dict means
    everything passed. `async` for call-site compatibility (callers already
    `await` this) even though the checks themselves are synchronous — no
    subprocess or I/O is involved."""
    field_texts: list[tuple[str, str, str]] = []  # (qid, field_label, text)
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "")
        field_texts.append((qid, "text", q.get("text") or ""))
        field_texts.append((qid, "explanation", q.get("explanation") or ""))
        field_texts.append((qid, "correct_answer", str(q.get("correct_answer") or "")))
        opts = q.get("options")
        if isinstance(opts, list):
            for oi, opt in enumerate(opts):
                field_texts.append((qid, f"option[{oi}]", option_text(opt)))

    issues: dict[str, list[str]] = {}

    for qid, field, text in field_texts:
        if has_unbalanced_dollar(text):
            issues.setdefault(qid, []).append(f"unbalanced $ delimiter in {field}: {text!r}")

        for seg in extract_math_segments(text):
            if has_unbalanced_braces(seg):
                issues.setdefault(qid, []).append(f"unbalanced braces in {field}: {seg!r}")
            elif looks_like_bare_chemical_formula(seg):
                issues.setdefault(qid, []).append(
                    f"chemical formula not wrapped in \\ce{{...}} in {field}: {seg!r}"
                )

    return issues
