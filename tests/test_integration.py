from unittest.mock import AsyncMock, Mock

import pytest

from mcp_web_tools import search as search_mod


@pytest.mark.asyncio
async def test_web_search_prefers_first_successful_provider(monkeypatch):
    brave_result = {
        "provider": "brave",
        "results": [
            {"title": "Brave Result", "url": "https://brave.test", "description": "primary"}
        ],
    }

    brave_mock = AsyncMock(return_value=brave_result)
    monkeypatch.setattr(search_mod, "brave_search", brave_mock)

    google_mock = Mock(return_value={"provider": "google", "results": []})
    duck_mock = Mock(return_value={"provider": "duckduckgo", "results": []})
    monkeypatch.setattr(search_mod, "google_search", google_mock)
    monkeypatch.setattr(search_mod, "duckduckgo_search", duck_mock)

    result = await search_mod.web_search("test query", limit=4)

    assert result == brave_result
    brave_mock.assert_awaited_once_with("test query", 4)
    google_mock.assert_not_called()
    duck_mock.assert_not_called()


@pytest.mark.asyncio
async def test_web_search_chains_fallbacks(monkeypatch):
    brave_mock = AsyncMock(return_value=None)
    monkeypatch.setattr(search_mod, "brave_search", brave_mock)

    google_mock = Mock(return_value=None)
    monkeypatch.setattr(search_mod, "google_search", google_mock)

    duck_result = {
        "provider": "duckduckgo",
        "results": [
            {"title": "Fallback", "url": "https://duck.test", "description": "last resort"}
        ],
    }
    duck_mock = Mock(return_value=duck_result)
    monkeypatch.setattr(search_mod, "duckduckgo_search", duck_mock)

    result = await search_mod.web_search("test query", limit=5)

    assert result == duck_result
    brave_mock.assert_awaited_once_with("test query", 5)
    google_mock.assert_called_once_with("test query", 5)
    duck_mock.assert_called_once_with("test query", 5)
