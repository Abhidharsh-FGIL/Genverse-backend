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

import logging
import re

_log = logging.getLogger(__name__)

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


# ─── Over-escaped backslash collapsing ───────────────────────────────────────
#
# An LLM asked to emit LaTeX inside a JSON string sometimes escapes the
# backslash a SECOND time, so "$\theta$" arrives as "$\\theta$" once the JSON
# is decoded. In TeX a doubled backslash is a LINE BREAK, so the damage is
# visible rather than subtle: "\\theta" renders as a line break followed by
# italic "theta", "5\\,\\mathrm{kg}" renders as "5", a line break, then a
# literal ", kg", and "37^\\circ" is an outright parse error that KaTeX shows
# as red source text. Those are exactly the three symptoms reported from the
# Assessment Hub.
#
# Collapsing has to be surgical, because a doubled backslash is ALSO valid,
# intentional LaTeX: it is the row separator inside matrix/cases/aligned
# blocks. Two independent guards keep genuine separators safe.
#
#   1. The character AFTER the run. A real "\\" separator is always followed
#      by whitespace, a digit, "[" (as in "\\[1em]"), "*" or the end of the
#      segment — never directly by a command letter. Every one of the 31
#      multi-backslash runs found in real stored assessment data is "\\"
#      followed by a space, inside a pmatrix/bmatrix. So the lookahead below
#      admits ONLY characters that can start a LaTeX command, a spacing macro
#      or a script marker, and digits/whitespace/"["/"*" are deliberately
#      excluded.
#   2. Whole \begin{...}...\end{...} environments are skipped verbatim, so a
#      separator written with no following space ("\begin{cases}x\\y\end{cases}")
#      survives guard 1 not applying to it. Nested environments are covered too,
#      because the outermost \begin/\end pair is what gets skipped.
#
# KNOWN, DELIBERATE LIMITATION: a genuine line break with NO following space and
# NO enclosing environment — "$$a\\b$$" — is collapsed to "$$a\b$$". Neither
# guard applies to it, and the character stream alone cannot distinguish it from
# an over-escaped "\b". The trade is made knowingly and in favour of collapsing,
# because:
#   - the shape is vanishingly rare (all 31 multi-backslash runs in real stored
#     assessment data are "\\" + a space, inside a pmatrix/bmatrix), whereas
#     over-escaped commands starting with a letter are the entire reported bug;
#   - refusing to collapse before a letter would leave "\\theta", "\\cos" and
#     "\\circ" broken, which is the thing this function exists to fix;
#   - a multi-row display block should be written as an environment anyway, and
#     any such block IS protected by guard 2.
# The behaviour is asserted in tests/test_latex_escaping.py so it stays a
# conscious decision rather than an accident.
#
# Anything outside a $...$ / $$...$$ math span is left completely alone: a
# backslash run in prose is far more likely to be a Windows path or a Python
# escape in a programming question than broken LaTeX.

# Environments whose "\\" is a genuine row/line separator.
_LINEBREAK_ENVS = (
    "cases", "aligned", "align", "alignat", "gathered", "gather", "split",
    "matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix",
    "smallmatrix", "array", "substack", "multline", "eqnarray",
)

# A single-backslash \begin{env}...\end{env} span. The (?<!\\) guards mean an
# already-over-escaped "\\begin{...}" does NOT match, so such a block is still
# handed to the collapser (it needs repairing like any other doubled command).
# The "(?:\\\\)*" before each \begin/\end absorbs an EVEN number of preceding
# backslashes, which is what tells a real command from an over-escaped one: in
# "x \\\\\end{aligned}" the \end is preceded by a complete "\\" line break (even,
# so \end is genuine and the environment must still be recognised), whereas in
# an over-escaped "\\end{cases}" it is preceded by a single stray backslash
# (odd, so this is not a real \end and the block SHOULD be collapsed). A plain
# (?<!\\) lookbehind cannot express that: it rejected both, so a trailing row
# separator silently disabled the environment guard around it.
_ENV_SPAN_RE = re.compile(
    r"(?<!\\)(?:\\\\)*\\begin\{(" + "|".join(_LINEBREAK_ENVS) + r")\*?\}"
    r"[\s\S]*?"
    r"(?<!\\)(?:\\\\)*\\end\{\1\*?\}"
)

# 2+ backslashes immediately followed by a character that can only begin a
# LaTeX command (letter), a spacing macro (\, \; \! \:) or a script marker
# (^ _). Whitespace, digits, "[" and "*" are excluded so genuine line breaks
# never match.
_OVER_ESCAPED_RE = re.compile(r"\\{2,}(?=[A-Za-z,;!:^_])")


def _collapse_in_math_segment(segment: str) -> str:
    """Collapse over-escaped commands inside ONE math segment, leaving any
    line-break environment (cases/aligned/pmatrix/...) byte-for-byte intact."""
    out: list[str] = []
    pos = 0
    for m in _ENV_SPAN_RE.finditer(segment):
        out.append(_OVER_ESCAPED_RE.sub(lambda _m: "\\", segment[pos:m.start()]))
        out.append(m.group(0))  # environment body preserved verbatim
        pos = m.end()
    out.append(_OVER_ESCAPED_RE.sub(lambda _m: "\\", segment[pos:]))
    return "".join(out)


# ─── Math-span protection: sentinels and SAFE restoration ────────────────────
#
# Several repairs below work only on the text OUTSIDE math spans, so the spans
# are swapped for sentinels, the repair runs, and the spans are put back.
#
# Restoring this naively destroys content, and did so in production. The two
# protection passes run in sequence — display math first, then inline — so an
# inline span can end up CONTAINING a display sentinel. That happens whenever
# the model writes two adjacent inline spans with no separator ("$^{10}$$\ce{B}$"):
# the "$$" in the middle is read as display math, matched greedily to the next
# "$$", and everything between is swallowed into one sentinel; the surrounding
# "$...$" is then protected as a second, OUTER sentinel that contains the first.
#
# Restoring forward with str.replace(..., 1) then silently fails: by the time
# the loop reaches the inner sentinel it is no longer in `result` at all (it is
# nested inside a later entry), so the replace is a no-op, and the outer
# restore re-inserts text that still carries the raw \x02...\x03 bytes. The
# stored question then contains literal control characters, KaTeX refuses it,
# and the swallowed text is gone for good.
#
# Restoring in REVERSE index order fixes the nesting (an outer sentinel always
# has a higher index than the inner one it contains), and the loop repeats
# until no sentinel remains in case of deeper nesting. `assert_no_sentinels`
# is the backstop: these functions must NEVER return text containing a
# sentinel, so a caller that cannot fully restore returns its input untouched
# rather than emitting corruption.
_SENTINEL_START = "\x02"
_SENTINEL_END = "\x03"


def contains_sentinel(text: str) -> bool:
    """True if `text` carries a raw protection sentinel. In stored content this
    always means a previous repair corrupted it (see above)."""
    return bool(text) and (_SENTINEL_START in text or _SENTINEL_END in text)


def _restore_protected(result: str, protected: list[str], prefix: str) -> str:
    """Put protected spans back, innermost-last, repeating until stable."""
    for _ in range(len(protected) + 1):
        if _SENTINEL_START not in result:
            break
        for i in range(len(protected) - 1, -1, -1):
            result = result.replace(f"{_SENTINEL_START}{prefix}{i}{_SENTINEL_END}",
                                    protected[i])
    return result


def collapse_over_escaped_commands(text: str) -> str:
    """Collapse "\\\\theta" -> "\\theta" (and \\, \\circ, \\cos, \\mathrm, ...)
    inside math spans only, preserving genuine LaTeX line breaks."""
    if not text or "\\\\" not in text:
        return text
    if contains_sentinel(text):
        # Already carries control characters from an earlier failed restore —
        # running sentinel-based protection over it would compound the damage.
        return text

    segments: list[str] = []

    def _protect(m: re.Match) -> str:
        segments.append(_collapse_in_math_segment(m.group(0)))
        return f"{_SENTINEL_START}C{len(segments) - 1}{_SENTINEL_END}"

    # Display math first, so its "$" characters can't be mistaken for inline
    # delimiters by the second pass (same ordering extract_math_segments uses).
    result = _DISPLAY_RE.sub(_protect, text)
    result = _INLINE_RE.sub(_protect, result)
    result = _restore_protected(result, segments, "C")
    if contains_sentinel(result):
        _log.error(
            "[LatexValidator] collapse_over_escaped_commands could not restore "
            "every protected span; returning the input unchanged rather than "
            "emitting control characters. input=%r", text[:200],
        )
        return text
    return result


def math_environment_spans(text: str) -> list[str]:
    r"""Every \begin{env}...\end{env} span in `text`, verbatim, for the
    line-break environments this module protects. Public because the migration
    script and its tests compare these before and after a transformation to
    prove no genuine "\\" row separator was destroyed."""
    if not text:
        return []
    return [m.group(0) for m in _ENV_SPAN_RE.finditer(text)]


def has_over_escaped_command(text: str) -> bool:
    """True if `text` still contains a doubled backslash before a LaTeX command
    character inside a math span — the invariant the regression test asserts is
    never true for stored content."""
    if not text or "\\\\" not in text:
        return False
    for seg in extract_math_segments(text):
        stripped = _ENV_SPAN_RE.sub("", seg)
        if _OVER_ESCAPED_RE.search(stripped):
            return True
    return False


def repair_common_latex_issues(text: str) -> str:
    """Safe, conservative repairs applied before validation/save — used by
    both the one-off DB migration script (scripts/migrate_latex_normalize.py)
    and finalize_generated_questions. Mirrors the frontend normalizer's
    fixes: unbalanced leading $, double-escaped macro backslashes, a stray
    extra "$" after a display-math close, and a stray \\% that sits outside
    any math span."""
    if not text:
        return text
    if contains_sentinel(text):
        # Already-corrupted content (control characters from an earlier failed
        # restore). Repairing it is not possible — the swallowed text is gone —
        # and protecting it again would only nest more sentinels. Leave it for
        # the validator to flag and a human to regenerate.
        _log.warning(
            "[LatexValidator] input already contains protection sentinels "
            "(previously corrupted); leaving it unchanged: %r", text[:200],
        )
        return text

    result = repair_unbalanced_leading_dollar(text)
    result = _collapse_stray_triple_dollar(result)
    # Collapse every over-escaped LaTeX command, not just the brace-form ones.
    # The previous rule here was re.sub(r"\\{2,}([a-zA-Z]+\{)", ...), which
    # required a "{" immediately after the command name — so it repaired
    # \\mathrm{kg} and \\frac{a}{b} but silently left \\theta, \\,,
    # \\circ, \\cos, \\sin and \\mu_s doubled. That is precisely why a
    # reported question rendered "\mathrm{kg}" correctly while the "\," right
    # in front of it still came out as a line break plus a stray comma.
    result = collapse_over_escaped_commands(result)

    protected: list[str] = []

    def _protect(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"{_SENTINEL_START}P{len(protected) - 1}{_SENTINEL_END}"

    result = _DISPLAY_RE.sub(_protect, result)
    result = _INLINE_RE.sub(_protect, result)
    # Whatever's left is outside any math span — a literal \% here is prose
    # that over-escaped a percent sign, not valid TeX.
    result = result.replace("\\%", "%")
    result = _restore_protected(result, protected, "P")
    if contains_sentinel(result):
        _log.error(
            "[LatexValidator] repair_common_latex_issues could not restore every "
            "protected span; returning the input unchanged rather than emitting "
            "control characters. input=%r", text[:200],
        )
        return text

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


# C0 control characters that must never appear in question content. \t, \n and
# \r are legitimate (a multi-line stem), everything else is corruption:
#   - \x02/\x03 are this module's own protection sentinels, left behind by a
#     failed restore (see _restore_protected).
#   - \x08, \x0c, \x0b are what json.loads() produces from a LaTeX command that
#     was under-escaped in the model's JSON ("\begin" -> backspace + "egin",
#     "\frac" -> form-feed + "rac").
# Either way the text is broken in a way KaTeX rejects, and this check finds it
# with no Node/katex dependency at all — which matters, because the structural
# checks are the only ones that still run when real KaTeX validation is
# unavailable, and they were previously blind to this entire class.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def has_control_characters(text: str) -> bool:
    """True if `text` contains a C0 control character other than tab/newline/CR."""
    return bool(text) and bool(_CONTROL_CHAR_RE.search(text))


def describe_control_characters(text: str) -> str:
    """Human-readable list of the offending code points, for the log/flag."""
    found = sorted({ord(c) for c in _CONTROL_CHAR_RE.findall(text or "")})
    return ", ".join(f"U+{cp:04X}" for cp in found)


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
        if has_control_characters(text):
            issues.setdefault(qid, []).append(
                f"control characters ({describe_control_characters(text)}) in {field} — "
                f"content is corrupted and text may have been lost; regenerate this "
                f"question rather than editing it: {text[:120]!r}"
            )

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
