import os
import logging
from unittest.mock import patch, AsyncMock

import pytest

from mcp_web_tools import search as search_mod


class TestProviderLogging:
    @pytest.mark.asyncio
    async def test_brave_search_logs_warning_on_failure(self, caplog):
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test"}):
            with patch("mcp_web_tools.search.BraveSearch") as mock_bs:
                instance = mock_bs.return_value
                instance.web = AsyncMock(side_effect=Exception("boom"))

                caplog.set_level(logging.WARNING, logger=search_mod.__name__)
                res = await search_mod.brave_search("q")

                assert res is None
                messages = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING]
                assert any("Brave search failed" in m for m in messages)

    def test_google_search_logs_warning_on_failure(self, caplog):
        with patch("googlesearch.search") as mock_search:
            mock_search.side_effect = Exception("fail")

            caplog.set_level(logging.WARNING, logger=search_mod.__name__)
            res = search_mod.google_search("q")

            assert res is None
            messages = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING]
            assert any("Google search failed" in m for m in messages)

    def test_duckduckgo_search_logs_warning_on_failure(self, caplog):
        with patch("mcp_web_tools.search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.side_effect = Exception("nope")

            caplog.set_level(logging.WARNING, logger=search_mod.__name__)
            res = search_mod.duckduckgo_search("q")

            assert res is None
            messages = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING]
            assert any("DuckDuckGo search failed" in m for m in messages)

