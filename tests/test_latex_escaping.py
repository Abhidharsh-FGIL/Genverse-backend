r"""Over-escaped-backslash collapsing: the Assessment Hub LaTeX corruption.

An LLM writing LaTeX into a JSON string sometimes escapes the backslash twice,
so "$\theta$" arrives as "$\\theta$". TeX reads "\\" as a LINE BREAK, which is
why the reported symptoms were a stray line break before italic "theta", a
line break plus a literal ", kg" in front of \mathrm{kg}, and red KaTeX error
text for "37^\\circ".

The negative tests matter as much as the positive ones: a doubled backslash is
ALSO legitimate LaTeX (the row separator in matrix/cases/aligned), and every
multi-backslash run in real stored assessment data is one of those.
"""
import pytest

from app.services.ai_service import AIService
from app.services.latex_validator import (
    collapse_over_escaped_commands,
    has_over_escaped_command,
    repair_common_latex_issues,
)


# ── Positive: over-escaped commands must collapse ────────────────────────────

@pytest.mark.parametrize("broken,fixed", [
    # The exact strings from the reported Q3/Q6/Q7/Q10 screenshots.
    (r"$\\theta$",                      r"$\theta$"),
    (r"$\theta = 37^\\circ$".replace(r"\theta", r"\\theta"),
                                        r"$\theta = 37^\circ$"),
    (r"$m = 5\\,\\mathrm{kg}$",         r"$m = 5\,\mathrm{kg}$"),
    (r"$\\mu_s$",                       r"$\mu_s$"),
    (r"$N = mg\\cos\\theta$",           r"$N = mg\cos\theta$"),
    (r"$\\sin^2\\theta + \\cos^2\\theta = 1$", r"$\sin^2\theta + \cos^2\theta = 1$"),
    (r"$\\frac{a}{b}$",                 r"$\frac{a}{b}$"),
    (r"$\\ce{H2SO4}$",                  r"$\ce{H2SO4}$"),
    (r"$3 \\times 10^8$",               r"$3 \times 10^8$"),
    (r"$\\tan\\theta$",                 r"$\tan\theta$"),
    (r"$\\sqrt{2}$",                    r"$\sqrt{2}$"),
    (r"$\\left( x \\right)$",           r"$\left( x \right)$"),
    # spacing macros
    (r"$a\;b\\!c\\:d$",                r"$a\;b\!c\:d$"),
    # script markers
    (r"$x^\\circ_\\alpha$",             r"$x^\circ_\alpha$"),
    # display math
    (r"$$\\int_0^\\infty e^{-x}\\,dx = 1$$", r"$$\int_0^\infty e^{-x}\,dx = 1$$"),
    # more than two backslashes still collapses to one
    (r"$\\\\theta$",                    r"$\theta$"),
    (r"$\\\theta$",                     r"$\theta$"),
])
def test_over_escaped_commands_collapse(broken, fixed):
    assert collapse_over_escaped_commands(broken) == fixed


def test_full_reported_q10_stem():
    broken = (r"A block rests on an incline at $\\theta = 37^\\circ$ with mass "
              r"$m = 5\\,\\mathrm{kg}$. The normal force is $N = mg\\cos\\theta$ "
              r"and $\\mu_s$ is ___.")
    fixed = (r"A block rests on an incline at $\theta = 37^\circ$ with mass "
             r"$m = 5\,\mathrm{kg}$. The normal force is $N = mg\cos\theta$ "
             r"and $\mu_s$ is ___.")
    assert collapse_over_escaped_commands(broken) == fixed


# ── Negative: genuine LaTeX line breaks must survive ─────────────────────────

@pytest.mark.parametrize("intact", [
    # Every multi-backslash run in real stored data has this exact shape.
    r"$A = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix}$",
    r"$\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$",
    r"$$\begin{pmatrix}1&2\\3&4\end{pmatrix}$$",
    # Separator directly followed by a LETTER — only the environment guard
    # saves this one, the lookahead alone would not.
    r"$\begin{cases} x & y \\ a & b \end{cases}$",
    r"$f(x) = \begin{cases} x \\ y \end{cases}$",
    r"$\begin{cases}x\\y\end{cases}$",
    r"$\begin{aligned} a &= b \\ c &= d \end{aligned}$",
    r"$\begin{aligned}a&=b\\c&=d\end{aligned}$",
    r"$\begin{array}{cc} 1 & 2 \\ 3 & 4 \end{array}$",
    r"$\begin{vmatrix} a & b \\ c & d \end{vmatrix}$",
    r"$\begin{gathered} x \\ y \end{gathered}$",
    r"$\begin{align*} u &= v \\ w &= z \end{align*}$",
    # Line break outside an environment: followed by whitespace / digit / "["
    r"$$a + b \\ c + d$$",
    r"$$x \\[1em] y$$",
    r"$$1 \\ 2$$",
    # Correctly single-escaped content is never touched
    r"$\theta = 37^\circ$",
    r"$m = 5\,\mathrm{kg}$",
    # No math at all
    "Plain prose with no math whatsoever.",
    "",
])
def test_genuine_latex_is_untouched(intact):
    assert collapse_over_escaped_commands(intact) == intact


def test_prose_backslashes_are_never_rewritten():
    """Outside a math span a backslash run is far more likely to be a Windows
    path or a Python escape in a programming question than broken LaTeX."""
    for prose in (
        r"The path is C:\\Users\\student\\notes.txt",
        r"In Python, '\\n' is a literal backslash followed by n.",
        r"Escape a backslash in a regex as \\\\.",
    ):
        assert collapse_over_escaped_commands(prose) == prose


def test_mixed_broken_command_and_intact_matrix():
    """A doubled command outside the environment is repaired; the environment's
    own row separator in the same string is not."""
    broken = r"Given $\\theta$, evaluate $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$."
    fixed = r"Given $\theta$, evaluate $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$."
    assert collapse_over_escaped_commands(broken) == fixed


# ── Idempotence ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    r"$\\theta = 37^\\circ$",
    r"$\begin{cases} x \\ y \end{cases}$",
    r"$m = 5\\,\\mathrm{kg}$",
])
def test_collapse_is_idempotent(text):
    once = collapse_over_escaped_commands(text)
    assert collapse_over_escaped_commands(once) == once


# ── The detector used by the regression assertion ────────────────────────────

def test_detector_flags_only_real_corruption():
    assert has_over_escaped_command(r"$\\theta$")
    assert has_over_escaped_command(r"$5\\,\\mathrm{kg}$")
    assert has_over_escaped_command(r"$37^\\circ$")
    assert not has_over_escaped_command(r"$\theta$")
    assert not has_over_escaped_command(r"$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$")
    assert not has_over_escaped_command(r"$\begin{cases}x\\y\end{cases}$")
    assert not has_over_escaped_command(r"$$a \\ b$$")
    assert not has_over_escaped_command("no math here")
    assert not has_over_escaped_command("")


# ── repair_common_latex_issues keeps its existing behaviour ──────────────────

def test_repair_still_handles_brace_form_and_now_the_rest():
    # the case the OLD rule already handled
    assert repair_common_latex_issues(r"$\\ce{NaHCO3}$") == r"$\ce{NaHCO3}$"
    assert repair_common_latex_issues(r"$\\mathrm{kg}$") == r"$\mathrm{kg}$"
    # the cases it silently missed, which is the bug under repair
    assert repair_common_latex_issues(r"$\\theta$") == r"$\theta$"
    assert repair_common_latex_issues(r"$5\\,\\mathrm{kg}$") == r"$5\,\mathrm{kg}$"
    assert repair_common_latex_issues(r"$37^\\circ$") == r"$37^\circ$"


def test_repair_preserves_unrelated_fixes():
    # unbalanced leading $ still gets closed
    assert repair_common_latex_issues(r"$15.90\%") == r"$15.90\%$"
    # matrix line break survives the whole repair pipeline
    src = r"$A = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix}$"
    assert repair_common_latex_issues(src) == src


# ── _fix_json_escapes: must no longer be add-only ────────────────────────────

def _decode(raw_json_text):
    import json
    return json.loads(AIService._fix_json_escapes(raw_json_text))


def test_json_escapes_repairs_single_backslash():
    """A lone "\t"-style LaTeX command is still promoted to a valid escape."""
    assert _decode(r'"$\theta$"') == r"$\theta$"
    assert _decode(r'"$\frac{a}{b}$"') == r"$\frac{a}{b}$"


def test_json_escapes_passes_correct_double_through():
    assert _decode(r'"$\\theta$"') == r"$\theta$"
    assert _decode(r'"$5\\,\\mathrm{kg}$"') == r"$5\,\mathrm{kg}$"


def test_json_escapes_collapses_over_escaped():
    """The regression: four backslashes in JSON text used to decode to a
    doubled backslash and reach storage untouched."""
    assert _decode(r'"$\\\\theta$"') == r"$\theta$"
    assert _decode(r'"$\\\\mu_s$"') == r"$\mu_s$"
    assert _decode(r'"$37^\\\\circ$"') == r"$37^\circ$"


def test_json_escapes_preserves_matrix_line_break():
    """In JSON text a genuine separator is also 4 backslashes — but it is
    followed by a space, not a command character."""
    assert _decode(r'"$\\begin{pmatrix}1&2\\\\ 3&4\\end{pmatrix}$"') == \
        r"$\begin{pmatrix}1&2\\ 3&4\end{pmatrix}$"
