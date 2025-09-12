import os
from pathlib import Path

import pytest

from mcp_web_tools.search import brave_search, google_search, duckduckgo_search


def _load_dotenv_if_present():
    """Lightweight .env loader without external deps.

    Only sets env vars that aren't already present.
    Supports simple KEY=VALUE lines and ignores comments/blank lines.
    """
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
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


@pytest.mark.asyncio
@pytest.mark.skipif(not RUN_LIVE, reason="Live search tests disabled; set RUN_LIVE_SEARCH_TESTS=1 to enable")
async def test_live_brave_search_non_empty():
    _load_dotenv_if_present()
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        pytest.skip("BRAVE_SEARCH_API_KEY not set; provide it in environment or .env")

    result = await brave_search("OpenAI", limit=3)
    assert result is not None
    assert result["provider"] == "brave"
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0


@pytest.mark.skipif(not RUN_LIVE, reason="Live search tests disabled; set RUN_LIVE_SEARCH_TESTS=1 to enable")
def test_live_google_search_non_empty():
    result = google_search("OpenAI", limit=3)
    assert result is not None
    assert result["provider"] == "google"
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0


@pytest.mark.skipif(not RUN_LIVE, reason="Live search tests disabled; set RUN_LIVE_SEARCH_TESTS=1 to enable")
def test_live_duckduckgo_search_non_empty():
    result = duckduckgo_search("OpenAI", limit=3)
    assert result is not None
    assert result["provider"] == "duckduckgo"
    assert isinstance(result["results"], list)
    assert len(result["results"]) > 0

