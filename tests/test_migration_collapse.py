r"""The migration's transformation logic, exercised WITHOUT a database.

scripts/migrate_collapse_over_escaped_latex.py does its work in pure functions
(collapse_question / all_text / _env_bodies) so the risky part — what it would
write — is testable offline. The database round trip is deliberately not
covered here; it is exercised by running the script itself.
"""
import pytest

from scripts.migrate_collapse_over_escaped_latex import (
    collapse_question, all_text, _env_bodies,
)


CORRUPT_QUESTION = {
    "id": "q10", "type": "match", "points": 2,
    "text": r"Incline $\\theta = 37^\\circ$, mass $m = 5\\,\\mathrm{kg}$.",
    "options": [r"$mg\\sin\\theta$", r"$\\mu_s N$"],
    "pairs": [{"left": r"Along incline", "right": r"$mg\\sin\\theta$"},
              {"left": r"Friction", "right": r"$\\mu_s N$"}],
    "correctAnswer": r"$mg\\cos\\theta$",
    "explanation": r"Uses $3 \\times 10^8$ and $\\frac{a}{b}$.",
}


def test_every_text_field_is_collapsed():
    out, changes = collapse_question(CORRUPT_QUESTION)
    fields = {c[0] for c in changes}
    assert fields == {
        "text", "correctAnswer", "explanation",
        "options[0]", "options[1]", "pairs[0].right", "pairs[1].right",
    }
    joined = " ".join(all_text(out))
    assert r"\\theta" not in joined
    assert r"\\," not in joined
    assert r"\\circ" not in joined
    assert r"\\mu_s" not in joined
    assert r"\theta" in joined and r"\mathrm{kg}" in joined


def test_both_field_spellings_are_handled():
    r"""Stored data carries correct_answer (generator) AND correctAnswer
    (frontend edit-save); a migration that knew only one would skip half."""
    for key in ("correct_answer", "correctAnswer"):
        q = {"id": "x", key: r"$\\theta$"}
        out, changes = collapse_question(q)
        assert changes, f"{key} was not migrated"
        assert out[key] == r"$\theta$"


def test_unchanged_question_reports_no_changes():
    clean = {"id": "c", "text": r"Angle $\theta = 37^\circ$", "options": [r"$\frac{1}{2}$"]}
    out, changes = collapse_question(clean)
    assert changes == []
    assert out == clean


def test_line_breaks_survive_and_environments_are_identical():
    q = {
        "id": "m", "text": r"$A = \begin{pmatrix} 2 & 3 \\ 1 & 4 \end{pmatrix}$",
        "options": [r"$\begin{cases}x\\y\end{cases}$"],
        "explanation": r"$$\begin{aligned} a &= b \\ c &= d \end{aligned}$$",
    }
    out, changes = collapse_question(q)
    assert changes == []
    before = [e for t in all_text(q) for e in _env_bodies(t)]
    after = [e for t in all_text(out) for e in _env_bodies(t)]
    assert sorted(before) == sorted(after)
    assert len(before) == 3


def test_mixed_question_repairs_command_but_not_separator():
    q = {"id": "mix",
         "text": r"Given $\\theta$, evaluate $\begin{pmatrix}1&2\\ 3&4\end{pmatrix}$."}
    out, changes = collapse_question(q)
    assert len(changes) == 1
    assert r"$\theta$" in out["text"]
    assert r"\\ 3&4" in out["text"], "matrix row separator must survive"


def test_non_dict_and_empty_inputs_are_safe():
    assert collapse_question("not a dict") == ("not a dict", [])
    assert collapse_question({}) == ({}, [])
    assert all_text({}) == []


def test_dict_shaped_options_are_supported():
    q = {"id": "d", "options": [{"text": r"$\\theta$", "id": "a"}]}
    out, changes = collapse_question(q)
    assert changes == [("options[0].text", r"$\\theta$", r"$\theta$")]
    assert out["options"][0]["id"] == "a", "non-text keys must be preserved"


def test_original_question_is_not_mutated():
    original = dict(CORRUPT_QUESTION)
    text_before = original["text"]
    collapse_question(CORRUPT_QUESTION)
    assert CORRUPT_QUESTION["text"] == text_before
