from unittest.mock import AsyncMock

import pytest

from mcp_web_tools.__init__ import fetch_url, search_web


@pytest.mark.asyncio
async def test_search_web_tool_delegates_to_web_search(monkeypatch):
    expected = {"provider": "brave", "results": []}
    mock_web_search = AsyncMock(return_value=expected)
    monkeypatch.setattr("mcp_web_tools.__init__.web_search", mock_web_search)

    result = await search_web("query", offset=2)

    assert result == expected
    mock_web_search.assert_awaited_once_with("query", 10, 2)


@pytest.mark.asyncio
async def test_fetch_url_tool_delegates_with_arguments(monkeypatch):
    mock_loader = AsyncMock(return_value="content")
    monkeypatch.setattr("mcp_web_tools.__init__.load_content", mock_loader)

    result = await fetch_url("https://example.com/page", offset=5, raw=True)

    assert result == "content"
    mock_loader.assert_awaited_once_with("https://example.com/page", 20_000, 5, True)


@pytest.mark.asyncio
async def test_fetch_url_tool_uses_default_arguments(monkeypatch):
    mock_loader = AsyncMock(return_value="default")
    monkeypatch.setattr("mcp_web_tools.__init__.load_content", mock_loader)

    result = await fetch_url("https://example.com/other", offset=0, raw=False)

    assert result == "default"
    mock_loader.assert_awaited_once_with("https://example.com/other", 20_000, 0, False)
