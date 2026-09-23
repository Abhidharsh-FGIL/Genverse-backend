"""Which revision is actually running, resolved once at import.

Exists so a deployed environment's code version can be checked from the
outside (GET /version) instead of guessed. Resolution order, first hit wins:

1. GIT_SHA / GIT_COMMIT / SOURCE_VERSION env var — what a container build or
   PaaS injects (Railway, Render, Heroku and Docker builds all set one of
   these), and the only thing available when the image has no .git directory.
2. A REVISION file written next to the app at build time.
3. `git rev-parse HEAD` in the source tree — the dev/bare-metal case.

Returns "unknown" rather than raising if none apply: a deployment should never
fail to boot because it cannot name itself.
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ENV_VARS = ("GIT_SHA", "GIT_COMMIT", "SOURCE_VERSION", "RENDER_GIT_COMMIT",
             "RAILWAY_GIT_COMMIT_SHA", "HEROKU_SLUG_COMMIT", "VERCEL_GIT_COMMIT_SHA")


@lru_cache(maxsize=1)
def git_sha() -> str:
    for var in _ENV_VARS:
        value = (os.environ.get(var) or "").strip()
        if value:
            return value[:40]

    revision_file = _ROOT / "REVISION"
    try:
        value = revision_file.read_text(encoding="utf-8").strip()
        if value:
            return value[:40]
    except OSError:
        pass

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT,
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()[:40]
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def build_info() -> dict:
    sha = git_sha()
    return {
        "git_sha": sha,
        "git_sha_short": sha[:7] if sha != "unknown" else "unknown",
        "environment": os.environ.get("ENVIRONMENT", "unknown"),
    }
