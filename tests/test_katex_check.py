r"""Real-KaTeX validation layer: parsing, availability reporting, and the
deliberate degradation when Node/katex is missing.

Self-contained: uses a local Node subprocess, never the network, never a
database. Skips (rather than fails) where Node or the katex package is absent,
because the whole point of this layer is that its absence must not break
anything.
"""
import asyncio

import pytest

from app.services import katex_check

pytestmark = pytest.mark.asyncio


async def _katex_available() -> bool:
    return await katex_check.check_segments([r"\theta"]) is not None


async def test_valid_math_parses():
    result = await katex_check.check_segments(
        [r"\theta", r"37^\circ", r"5\,\mathrm{kg}", r"\frac{a}{b}",
         r"\begin{pmatrix}1&2\\ 3&4\end{pmatrix}", r"\mu_s", r"mg\cos\theta"]
    )
    if result is None:
        pytest.skip("node/katex unavailable")
    assert result == {}


async def test_chemistry_needs_mhchem_and_gets_it():
    r"""Without the mhchem extension every \ce{...} the chemistry prompt is
    told to emit would be a false failure."""
    result = await katex_check.check_segments([r"\ce{H2SO4}", r"\ce{Na2CO3 + 2HCl -> 2NaCl + H2O + CO2}"])
    if result is None:
        pytest.skip("node/katex unavailable")
    assert result == {}


async def test_genuinely_broken_math_is_caught():
    result = await katex_check.check_segments([r"\frac{1}{2", r"37^\\circ"])
    if result is None:
        pytest.skip("node/katex unavailable")
    assert set(result) == {r"\frac{1}{2", r"37^\\circ"}


async def test_over_escaped_theta_parses_but_is_caught_by_the_other_check():
    r"""The complementarity the design depends on: "$\\theta$" is VALID KaTeX
    (a line break then the word theta), so only the structural detector sees
    it, while "37^\\circ" is a hard parse error only KaTeX sees."""
    from app.services.latex_validator import has_over_escaped_command
    result = await katex_check.check_segments([r"\\theta"])
    if result is None:
        pytest.skip("node/katex unavailable")
    assert result == {}                                  # KaTeX is happy
    assert has_over_escaped_command(r"$\\theta$")        # the other check is not


async def test_empty_and_whitespace_segments_are_ignored():
    assert await katex_check.check_segments([]) == {}
    assert await katex_check.check_segments(["", "   "]) == {}


async def test_missing_node_degrades_to_none_and_logs_warning(monkeypatch, caplog):
    """Generation must keep working when the checker cannot run."""
    monkeypatch.setenv("NODE_BINARY", "/nonexistent/node-binary")
    monkeypatch.setattr(katex_check, "_UNAVAILABLE_LOGGED", False)
    monkeypatch.setitem(katex_check._STATUS, "available", None)

    with caplog.at_level("WARNING"):
        result = await katex_check.check_segments([r"\theta"])

    assert result is None, "must return None (unknown), not raise"
    assert any("UNAVAILABLE" in r.message or "UNAVAILABLE" in r.getMessage()
               for r in caplog.records), "a WARNING must be emitted"
    assert katex_check.status()["katex_validation"] == "unavailable"


async def test_status_reports_active_after_successful_probe(monkeypatch):
    monkeypatch.setitem(katex_check._STATUS, "available", None)
    status = await katex_check.probe()
    if status["katex_validation"] == "unavailable":
        pytest.skip("node/katex unavailable")
    assert status["katex_validation"] == "active"
    assert katex_check.status()["katex_validation"] == "active"


async def test_probe_never_raises(monkeypatch):
    async def boom(_segments):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(katex_check, "check_segments", boom)
    monkeypatch.setitem(katex_check._STATUS, "available", None)
    status = await katex_check.probe()
    assert status["katex_validation"] == "unavailable"


async def test_question_field_issues_are_attributed_to_the_right_field():
    questions = [
        {"id": "good", "text": r"Angle $\theta = 37^\circ$", "explanation": r"$5\,\mathrm{kg}$",
         "correct_answer": "", "options": []},
        {"id": "bad", "text": r"Broken $\frac{1}{2$ here", "explanation": "",
         "correct_answer": "", "options": []},
    ]
    issues = await katex_check.check_question_fields(questions)
    if not issues and await katex_check.check_segments([r"\frac{1}{2"]) is None:
        pytest.skip("node/katex unavailable")
    assert "good" not in issues
    assert "bad" in issues
    assert any("text" in m for m in issues["bad"])


# ── fill-in-the-blank placeholders must sit OUTSIDE math (the q17 bug) ────────

BLANK_INSIDE_MATH = [
    r"4^2 = ___",              # the exact segment from the real q17
    r"m = ___\,\mathrm{kg}",
    r"\theta = ___^\circ",
]
BLANK_OUTSIDE_MATH = [
    r"4^2 = ",
    r"\mathrm{kg}",
    r"\theta",
]


@pytest.mark.parametrize("segment", BLANK_INSIDE_MATH)
async def test_blank_inside_math_is_a_katex_error(segment):
    r""""___" is not valid LaTeX: the underscore is a subscript operator with
    nothing to subscript. This is what broke a real stored question
    ($4^2 = ___$) and what prompt rule 14 exists to prevent."""
    result = await katex_check.check_segments([segment])
    if result is None:
        pytest.skip("node/katex unavailable")
    assert segment in result
    assert "_" in result[segment] or "subscript" in result[segment].lower() \
        or "Expected group" in result[segment]


@pytest.mark.parametrize("segment", BLANK_OUTSIDE_MATH)
async def test_same_content_parses_once_the_blank_is_outside(segment):
    result = await katex_check.check_segments([segment])
    if result is None:
        pytest.skip("node/katex unavailable")
    assert result == {}


async def test_question_with_blank_inside_math_is_flagged_for_review():
    r"""End to end: the generation pipeline must flag, not silently save, a
    question shaped like the real q17."""
    from app.services.ai_service import AIService
    raw = [{"id": "q17", "type": "fill", "marks": 1,
            "text": r"Compute $4^2 = ___$ and state the value.",
            "correct_answer": "16", "explanation": ""}]
    qj, _ = await AIService.finalize_generated_questions(raw, {"fill"})
    if not qj[0].get("needs_review") and await katex_check.check_segments([r"4^2 = ___"]) is None:
        pytest.skip("node/katex unavailable")
    assert qj[0]["needs_review"] is True
    assert "KaTeX" in qj[0]["review_reason"]


async def test_correctly_placed_blank_is_not_flagged():
    from app.services.ai_service import AIService
    raw = [{"id": "ok", "type": "fill", "marks": 1,
            "text": r"Compute $4^2 = $ ___ and state the value in $\mathrm{kg}$.",
            "correct_answer": "16", "explanation": ""}]
    qj, _ = await AIService.finalize_generated_questions(raw, {"fill"})
    assert not qj[0].get("needs_review"), qj[0].get("review_reason")
