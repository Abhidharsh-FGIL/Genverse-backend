r"""The protect/restore corruption that reached production.

repair_common_latex_issues swaps math spans for \x02P<n>\x03 sentinels, repairs
the prose around them, then puts them back. When the model wrote two adjacent
inline spans with no separator ("$^{10}$$\ce{B}$"), the "$$" in the middle was
read as display math and matched greedily to the next "$$", swallowing the text
between into one sentinel; the surrounding "$...$" was then protected as a
second sentinel CONTAINING the first. Restoring forward with replace(..., 1)
silently no-opped on the inner one, so the stored question ended up with literal
control characters and permanently lost text:

    stored: '$^{10}\x02P0\x03\ce{B}$'   <- \u0002P0\u0003 is our own sentinel

These tests pin the fix (reverse-order, repeat-until-stable restoration plus a
fail-safe) and the detection that now catches this class without needing Node.
"""
import pytest

from app.services.latex_validator import (
    collapse_over_escaped_commands, contains_sentinel, describe_control_characters,
    has_control_characters, repair_common_latex_issues, validate_questions_latex,
)

# The real generation output that produced the reported Chemistry Q7.
BORON_STEM = (
    r"Naturally occurring boron consists of two isotopes, $^{10}$$\ce{B}$ "
    r"(atomic mass = $10.01\,\mathrm{u}$) and $^{11}$$\ce{B}$ "
    r"(atomic mass = $11.01\,\mathrm{u}$). The average atomic mass of boron is "
    r"found to be $10.81\,\mathrm{u}$."
)


def test_the_exact_production_input_round_trips_losslessly():
    out = repair_common_latex_issues(BORON_STEM)
    assert not contains_sentinel(out), f"sentinel survived: {out!r}"
    assert out == BORON_STEM, "content was altered or lost"
    assert "^{11}" in out, "the second isotope must not be swallowed"


@pytest.mark.parametrize("src", [
    r"$x$$y$$z$",
    r"$a$$b$",
    r"boron $^{10}$$\ce{B}$ and $^{11}$$\ce{B}$ here",
    r"$^{10}$$\ce{B}$",
    r"$a$$b$$c$$d$",
    r"$$display$$ then $inline$",
    r"$inline$ then $$display$$",
    r"text $a$ more $b$ end",
])
def test_adjacent_and_nested_spans_never_strand_a_sentinel(src):
    out = repair_common_latex_issues(src)
    assert not contains_sentinel(out), f"{src!r} -> {out!r}"


@pytest.mark.parametrize("src", [
    r"$x$$y$$z$",
    r"boron $^{10}$$\\ce{B}$ and $^{11}$$\\ce{B}$ here",
    r"$^{10}$$\\theta$",
])
def test_collapse_over_escaped_commands_also_never_strands_a_sentinel(src):
    r"""The same flawed pattern was copied into the over-escape collapser."""
    out = collapse_over_escaped_commands(src)
    assert not contains_sentinel(out), f"{src!r} -> {out!r}"


def test_already_corrupted_input_is_left_alone_not_compounded():
    r"""Text that already carries sentinels cannot be repaired — the swallowed
    content is gone — and protecting it again would nest more sentinels."""
    corrupted = "isotopes, $^{10}\x02P0\x03\\ce{B}$ (mass = $11.01\\,\\mathrm{u}$)."
    assert repair_common_latex_issues(corrupted) == corrupted
    assert collapse_over_escaped_commands(corrupted) == corrupted


def test_repair_still_does_its_actual_job():
    """The fix must not disable the repairs this function exists for."""
    assert repair_common_latex_issues(r"$\\theta$") == r"$\theta$"
    assert repair_common_latex_issues(r"$5\\,\\mathrm{kg}$") == r"$5\,\mathrm{kg}$"
    assert repair_common_latex_issues(r"$15.90\%") == r"$15.90\%$"
    matrix = r"$A = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix}$"
    assert repair_common_latex_issues(matrix) == matrix


# ── detection: works with no Node/katex at all ───────────────────────────────

def test_control_characters_are_detected():
    assert has_control_characters("a\x02P0\x03b")
    assert has_control_characters("\x08egin{matrix}")      # \b from a JSON escape
    assert has_control_characters("\x0crac{1}{2}")         # \f from a JSON escape
    assert not has_control_characters("normal $\\theta$ text")
    assert not has_control_characters("multi\nline\ttext\r\n")
    assert not has_control_characters("")


def test_control_character_description_is_useful():
    assert describe_control_characters("a\x02b\x03c") == "U+0002, U+0003"


@pytest.mark.asyncio
async def test_validator_flags_the_stored_corrupted_question_without_katex():
    r"""The structural checks are all that run when real KaTeX validation is
    unavailable, and they were previously blind to this entire class — which is
    why the corrupted question saved silently."""
    stored = ("Naturally occurring boron consists of two isotopes, "
              "$^{10}\x02P0\x03\\ce{B}$ (atomic mass = $11.01\\,\\mathrm{u}$).")
    issues = await validate_questions_latex(
        [{"id": "q7", "text": stored, "explanation": "", "correct_answer": ""}]
    )
    assert "q7" in issues
    assert any("control characters" in m for m in issues["q7"])
    assert any("U+0002" in m for m in issues["q7"])
    assert any("regenerate" in m for m in issues["q7"]), "must say it needs regeneration"


# ── the retry recovers the real corrupted question (d) ───────────────────────

CORRUPTED_Q7 = {
    "id": "q7", "type": "short", "points": 2, "options": None,
    "text": ("Naturally occurring boron consists of two isotopes, "
             "$^{10}\x02P0\x03\\ce{B}$ (atomic mass = $11.01\\,\\mathrm{u}$). The average "
             "atomic mass of boron is found to be $10.81\\,\\mathrm{u}$. Analyze this data "
             "to calculate the percentage abundance of each isotope."),
    "explanation": ("Letting the fraction of $^{10}\\ce{B}$ be $x$ and $^{11}\\ce{B}$ be "
                    "$1-x$, the equation is $10.01x + 11.01(1-x) = 10.81$."),
    "correctAnswer": "The abundance of $^{10}\\ce{B}$ is $20\\%$ and $^{11}\\ce{B}$ is $80\\%$.",
}

# What a live model actually returned for this input, recorded so the test is
# deterministic and needs no network. Note it reconstructed the swallowed
# "$^{11}\ce{B}$ (atomic mass = ...)" from the explanation, which is exactly
# what the repair prompt asks for.
RECORDED_MODEL_FIX = {
    "id": "q7", "type": "short", "points": 2, "options": None,
    "text": ("Naturally occurring boron consists of two isotopes, $^{10}\\ce{B}$ "
             "(atomic mass = $10.01\\,\\mathrm{u}$) and $^{11}\\ce{B}$ (atomic mass = "
             "$11.01\\,\\mathrm{u}$). The average atomic mass of boron is found to be "
             "$10.81\\,\\mathrm{u}$. Analyze this data to calculate the percentage "
             "abundance of each isotope."),
    "explanation": CORRUPTED_Q7["explanation"],
    "correctAnswer": CORRUPTED_Q7["correctAnswer"],
}


@pytest.mark.asyncio
async def test_corrupted_question_is_flagged_then_recovered_by_the_retry(monkeypatch):
    import json as _json
    from app.services.ai_service import AIService
    from app.services.question_validation import validate_and_flag

    questions, stats = await validate_and_flag([dict(CORRUPTED_Q7)], context="test")
    assert stats["flagged"] == 1
    assert stats["control_char_questions"] == 1
    assert questions[0]["needs_review"] is True

    sent = []

    async def fake_chat(self, messages, **kwargs):
        sent.append(messages[0]["content"])
        return _json.dumps([RECORDED_MODEL_FIX])

    monkeypatch.setattr(AIService, "chat", fake_chat)
    out = await AIService().repair_flagged_raw_questions(questions)

    assert not out[0].get("needs_review"), out[0].get("review_reason")
    assert not has_control_characters(out[0]["text"])
    assert "^{11}" in out[0]["text"], "the swallowed isotope should be restored"
    # the prompt must actually tell the model how to handle this shape
    assert "control characters" in sent[0]
    assert "two math spans next to each other" in sent[0]


@pytest.mark.asyncio
async def test_retry_keeps_the_flag_when_the_model_cannot_fix_it(monkeypatch):
    import json as _json
    from app.services.ai_service import AIService
    from app.services.question_validation import validate_and_flag

    questions, _ = await validate_and_flag([dict(CORRUPTED_Q7)], context="test")

    async def unhelpful(self, messages, **kwargs):
        return _json.dumps([dict(CORRUPTED_Q7)])   # returns it still broken

    monkeypatch.setattr(AIService, "chat", unhelpful)
    out = await AIService().repair_flagged_raw_questions(questions, max_attempts=2)
    assert len(out) == 1, "must be kept, not dropped"
    assert out[0]["needs_review"] is True


# ── the marker is now visible, and must still never leak ─────────────────────

from app.services.latex_validator import _sentinel, _SENTINEL_OPEN  # noqa: E402

REPORTED_STEM = (
    r"Naturally occurring boron consists of two isotopes, $^{10}$$\ce{B}$ "
    r"(atomic mass = $10.01\,\mathrm{u}$) and $^{11}$$\ce{B}$ "
    r"(atomic mass = $11.01\,\mathrm{u}$)."
)

# Anything a C0 control character check would catch, for the "never emit an
# invisible byte" assertion.
import re as _re  # noqa: E402
_CONTROL_RE = _re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def test_the_marker_is_visible_not_a_control_character():
    r"""A leak must be obvious in a diff, a log and on screen — the whole
    reason the old \x02/\x03 scheme hid this bug for so long."""
    marker = _sentinel("P", 0)
    assert marker == "⟦PROTECT:P0⟧"
    assert not _CONTROL_RE.search(marker)
    assert marker.isprintable()


@pytest.mark.parametrize("src", [
    REPORTED_STEM,
    r"$^{10}$$\ce{B}$",
    r"$^{10}\ce{B}$ and $^{11}\ce{B}$",
    r"$\ce{H2SO4}$ reacts with $\ce{NaHCO3}$",
    r"$\frac{\ce{H2O}}{\ce{CO2}}$",
    r"$a$$b$$c$$d$",
    r"$A = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix}$",
    r"$f(x) = \begin{cases} x^2 & x \ge 0 \\ -x & x < 0 \end{cases}$",
    r"50% of $\ce{NaCl}$ and $x^2$",
    "plain prose, no math at all",
])
def test_no_marker_and_no_control_character_ever_reaches_the_output(src):
    for fn in (repair_common_latex_issues, collapse_over_escaped_commands):
        out = fn(src)
        assert _SENTINEL_OPEN not in out, f"{fn.__name__} leaked a marker: {out!r}"
        assert not contains_sentinel(out), f"{fn.__name__} leaked: {out!r}"
        assert not _CONTROL_RE.search(out), f"{fn.__name__} emitted a control char: {out!r}"


def test_the_reported_stem_keeps_both_isotopes():
    out = repair_common_latex_issues(REPORTED_STEM)
    assert "^{11}" in out, "the second isotope must not be swallowed"
    assert "10.01" in out and "11.01" in out
    assert out == REPORTED_STEM


def test_detection_covers_both_the_new_and_legacy_marker_forms():
    assert contains_sentinel("a⟦PROTECT:P0⟧b")
    assert contains_sentinel("a\x02P0\x03b")          # legacy, still in stored data
    assert not contains_sentinel(r"$\theta$ normal text")
    assert not contains_sentinel("")
