"""Real KaTeX parse validation for AI-generated math.

The structural checks in latex_validator.py (balanced $, balanced braces) are
pure Python and catch the common shapes, but they cannot tell you whether KaTeX
itself will actually accept a segment. This module closes that gap by running
the genuine KaTeX parser — the same library and the same mhchem extension the
frontend renders with — over every math segment before a generated assessment
is saved.

It shells out to Node once per batch (not once per segment) and degrades
gracefully: if Node or the KaTeX package is unavailable, `check_segments`
returns None, meaning "unknown", and callers fall back to the structural checks
alone rather than blocking a save. The failure is logged once per process so a
misconfigured deployment is visible without flooding the logs.

Note that a KaTeX parse check ALONE is not sufficient for the over-escaped
backslash bug: "$\\theta$" parses without error (as a line break followed by
the word "theta") even though it renders wrong. That shape is caught by
latex_validator.has_over_escaped_command, and the two checks are complementary
— "37^\\circ" is the reverse case, a hard parse error KaTeX does catch.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

_JS = Path(__file__).parent / "js" / "katex_check.js"
_UNAVAILABLE_LOGGED = False
_TIMEOUT_SECONDS = 30
_log = logging.getLogger(__name__)

# Whether real KaTeX validation is usable, resolved by probe() and reported on
# /health. None means "not probed yet". This is deliberately observable state:
# the fallback to structural-only checks is silent by design (generation must
# never break because a validator is missing), and silent degradation that
# nobody can see is how a weakened safety net goes unnoticed for months.
_STATUS: dict = {"available": None, "reason": "not probed"}


def status() -> dict:
    """Current KaTeX validation status, for /health."""
    available = _STATUS["available"]
    return {
        "katex_validation": (
            "active" if available else ("unavailable" if available is False else "unknown")
        ),
        "katex_validation_detail": _STATUS["reason"],
    }


def _mark_unavailable(reason: str) -> None:
    _STATUS.update(available=False, reason=reason)


def _mark_available() -> None:
    _STATUS.update(available=True, reason="real KaTeX parser in use")


def _log_unavailable(reason: str) -> None:
    global _UNAVAILABLE_LOGGED
    _mark_unavailable(reason)
    if not _UNAVAILABLE_LOGGED:
        _UNAVAILABLE_LOGGED = True
        _log.warning(
            "[KatexCheck] real KaTeX validation UNAVAILABLE (%s). Generated "
            "assessments will be checked with structural rules only, so a "
            "question whose math fails a real KaTeX parse can still be saved. "
            "Install Node and the katex package, or set KATEX_MODULE_PATH, to "
            "enable it. Generation itself is unaffected.",
            reason,
        )


async def probe() -> dict:
    """Run one throwaway check so the KaTeX status is known before any
    assessment is generated, instead of being discovered mid-save. Called at
    startup; never raises, and never blocks startup on failure."""
    try:
        result = await check_segments([r"\theta"])
    except Exception as e:  # pragma: no cover - defensive
        _log_unavailable(f"probe raised {type(e).__name__}: {e}")
        return status()
    if result is None:
        # check_segments already logged the specific reason.
        return status()
    _mark_available()
    _log.info("[KatexCheck] real KaTeX validation active")
    return status()


async def check_segments(segments: list[str]) -> dict[str, str] | None:
    """Parse every segment with real KaTeX.

    Returns {segment: error_message} for the ones that fail (an empty dict means
    everything parsed), or None if the checker could not be run at all.
    """
    segments = [s for s in dict.fromkeys(segments) if s and s.strip()]
    if not segments:
        return {}
    if not _JS.exists():
        _log_unavailable(f"missing {_JS}")
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            os.environ.get("NODE_BINARY", "node"), str(_JS),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError) as e:
        _log_unavailable(f"cannot start node: {e}")
        return None

    payload = json.dumps({"segments": segments}).encode()
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(payload), timeout=_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        proc.kill()
        _log_unavailable("node timed out")
        return None

    if proc.returncode != 0:
        _log_unavailable(f"node exited {proc.returncode}: {stderr.decode()[:200]}")
        return None

    try:
        result = json.loads(stdout.decode() or "{}")
    except json.JSONDecodeError:
        _log_unavailable("unparseable checker output")
        return None

    if not result.get("ok"):
        _log_unavailable(result.get("reason", "unknown"))
        return None
    _mark_available()
    return result.get("errors") or {}


async def check_question_fields(questions: list[dict]) -> dict[str, list[str]]:
    """Run the real KaTeX parser over every math segment of every question.

    Returns {question_id: [issue, ...]}. An empty dict means everything parsed
    (or that the checker was unavailable — see check_segments).
    """
    from app.services.latex_validator import extract_math_segments, option_text

    seg_owners: dict[str, list[tuple[str, str]]] = {}
    for q in questions:
        if not isinstance(q, dict):
            continue
        qid = str(q.get("id") or "")
        texts = [
            ("text", q.get("text") or ""),
            ("explanation", q.get("explanation") or ""),
            ("correct_answer", str(q.get("correct_answer") or "")),
        ]
        opts = q.get("options")
        if isinstance(opts, list):
            texts += [(f"option[{i}]", option_text(o)) for i, o in enumerate(opts)]
        for field, text in texts:
            for seg in extract_math_segments(text):
                seg_owners.setdefault(seg, []).append((qid, field))

    errors = await check_segments(list(seg_owners))
    if not errors:
        return {}

    issues: dict[str, list[str]] = {}
    for seg, msg in errors.items():
        for qid, field in seg_owners.get(seg, []):
            issues.setdefault(qid, []).append(
                f"KaTeX parse error in {field}: {msg} (segment: {seg!r})"
            )
    return issues
