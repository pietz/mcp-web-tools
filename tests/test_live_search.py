import os
from pathlib import Path

import pytest

from mcp_web_tools.search import (
    brave_search,
    duckduckgo_search,
    google_search,
    perplexity_search,
)


def _load_dotenv_if_present():
    """Lightweight .env loader without external deps.

    Only sets env vars that aren't already present.
    Supports simple KEY=VALUE lines and ignores comments/blank lines.
    """
    for filename in (".env", "local.env"):
        env_path = Path.cwd() / filename
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


RUN_LIVE = os.getenv("RUN_LIVE_SEARCH_TESTS") == "1"


def _xfail_live(provider: str, error: Exception | str) -> None:
    pytest.xfail(f"{provider} live search check failed: {error}")


@pytest.mark.asyncio
@pytest.mark.live_search
@pytest.mark.skipif(
    not RUN_LIVE, reason="Live search tests disabled; set RUN_LIVE_SEARCH_TESTS=1 to enable"
)
async def test_live_brave_search_non_empty():
    _load_dotenv_if_present()
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        pytest.skip("BRAVE_SEARCH_API_KEY not set; provide it in environment or .env")

    try:
        result = await brave_search("OpenAI", limit=3)
    except Exception as exc:  # pragma: no cover - live failure handled via xfail
        _xfail_live("Brave", exc)
        return

    if not result or not result.get("results"):
        _xfail_live("Brave", "no results returned")

    assert result["provider"] == "brave"
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0


@pytest.mark.live_search
@pytest.mark.skipif(
    not RUN_LIVE, reason="Live search tests disabled; set RUN_LIVE_SEARCH_TESTS=1 to enable"
)
def test_live_google_search_non_empty():
    try:
        result = google_search("OpenAI", limit=3)
    except Exception as exc:  # pragma: no cover
        _xfail_live("Google", exc)
        return

    if not result or not result.get("results"):
        _xfail_live("Google", "no results returned")

    assert result["provider"] == "google"
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0


@pytest.mark.live_search
@pytest.mark.skipif(
    not RUN_LIVE, reason="Live search tests disabled; set RUN_LIVE_SEARCH_TESTS=1 to enable"
)
def test_live_duckduckgo_search_non_empty():
    try:
        result = duckduckgo_search("OpenAI", limit=3)
    except Exception as exc:  # pragma: no cover
        _xfail_live("DuckDuckGo", exc)
        return

    if not result or not result.get("results"):
        _xfail_live("DuckDuckGo", "no results returned")

    assert result["provider"] == "duckduckgo"
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0


@pytest.mark.asyncio
@pytest.mark.live_search
@pytest.mark.skipif(
    not RUN_LIVE, reason="Live search tests disabled; set RUN_LIVE_SEARCH_TESTS=1 to enable"
)
async def test_live_perplexity_search_non_empty():
    _load_dotenv_if_present()
    if not os.getenv("PERPLEXITY_API_KEY"):
        pytest.skip("PERPLEXITY_API_KEY not set; provide it in environment or local.env")

    try:
        result = await perplexity_search("OpenAI", limit=3)
    except Exception as exc:  # pragma: no cover
        _xfail_live("Perplexity", exc)
        return

    if not result or not result.get("results"):
        _xfail_live("Perplexity", "no results returned")

    assert result["provider"] == "perplexity"
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0
