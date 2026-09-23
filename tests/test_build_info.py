r"""Revision reporting. No database, no network, no subprocess dependency:
the git fallback is only reached when no env var or REVISION file applies."""
import pytest

from app import build_info as bi


@pytest.fixture(autouse=True)
def _clear_caches():
    bi.git_sha.cache_clear()
    bi.build_info.cache_clear()
    yield
    bi.git_sha.cache_clear()
    bi.build_info.cache_clear()


@pytest.mark.parametrize("var", ["GIT_SHA", "GIT_COMMIT", "SOURCE_VERSION"])
def test_env_var_wins(monkeypatch, var):
    monkeypatch.setenv(var, "abc123def456")
    assert bi.git_sha() == "abc123def456"
    assert bi.build_info()["git_sha_short"] == "abc123d"


def test_revision_file_is_used_when_no_env_var(monkeypatch, tmp_path):
    for v in bi._ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    (tmp_path / "REVISION").write_text("f00dcafe\n")
    monkeypatch.setattr(bi, "_ROOT", tmp_path)
    assert bi.git_sha() == "f00dcafe"


def test_unknown_when_nothing_available(monkeypatch, tmp_path):
    for v in bi._ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(bi, "_ROOT", tmp_path)          # no REVISION, no .git
    def no_git(*a, **k):
        raise FileNotFoundError("git not installed")
    monkeypatch.setattr(bi.subprocess, "check_output", no_git)
    assert bi.git_sha() == "unknown"
    assert bi.build_info()["git_sha_short"] == "unknown"


def test_build_info_shape():
    info = bi.build_info()
    assert set(info) == {"git_sha", "git_sha_short", "environment"}


# ── /health and /version expose the state operators need ─────────────────────

@pytest.mark.asyncio
async def test_health_reports_revision_and_katex_status():
    r"""Calls the route function directly: no server, no database, no lifespan."""
    from app.main import health_check
    body = await health_check()
    assert body["status"] == "healthy"
    assert "git_sha" in body and "git_sha_short" in body
    assert body["katex_validation"] in {"active", "unavailable", "unknown"}
    assert "katex_validation_detail" in body


@pytest.mark.asyncio
async def test_version_endpoint_shape():
    from app.main import version
    body = await version()
    for key in ("service", "version", "git_sha", "git_sha_short", "environment"):
        assert key in body, f"/version missing {key}"
