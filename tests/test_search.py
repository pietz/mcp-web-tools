import logging
import os
from unittest.mock import patch, AsyncMock, Mock, call
from types import SimpleNamespace

import httpx
import pytest

from perplexity import PerplexityError
from perplexity.types.search_create_response import Result

from mcp_web_tools.search import (
    brave_search,
    duckduckgo_search,
    google_search,
    perplexity_search,
    web_search,
)


class TestBraveSearch:
    @pytest.mark.asyncio
    async def test_brave_search_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            result = await brave_search("test query")
            assert result is None

    @pytest.mark.asyncio
    async def test_brave_search_success(self):
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test_key"}):
            # Mock httpx AsyncClient context manager and response
            class MockResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "web": {
                            "results": [
                                {
                                    "title": "Test Title",
                                    "url": "https://test.com",
                                    "description": "Test Description",
                                }
                            ]
                        }
                    }

            class MockClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def get(self, *args, **kwargs):
                    return MockResponse()

            with patch("mcp_web_tools.search.httpx.AsyncClient", return_value=MockClient()):
                result = await brave_search("test query", limit=1)

            assert result is not None
            assert result["provider"] == "brave"
            assert len(result["results"]) == 1
            assert result["results"][0]["title"] == "Test Title"
            assert result["results"][0]["url"] == "https://test.com"
            assert result["results"][0]["description"] == "Test Description"

    @pytest.mark.asyncio
    async def test_brave_search_http_error(self):
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test_key"}):
            req = httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search")
            resp = httpx.Response(500, request=req)

            class MockResponse:
                def raise_for_status(self):
                    raise httpx.HTTPStatusError("error", request=req, response=resp)

                def json(self):
                    return {}

            class MockClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def get(self, *args, **kwargs):
                    return MockResponse()

            with patch("mcp_web_tools.search.httpx.AsyncClient", return_value=MockClient()):
                result = await brave_search("test query")

            assert result is None

    @pytest.mark.asyncio
    async def test_brave_search_retries_on_rate_limit(self, caplog):
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test_key"}):
            req = httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search")
            resp_429 = httpx.Response(429, request=req)

            class RateLimitedResponse:
                def raise_for_status(self):
                    raise httpx.HTTPStatusError("429", request=req, response=resp_429)

                def json(self):
                    return {}

            success_payload = {
                "web": {
                    "results": [
                        {
                            "title": "Recovered",
                            "url": "https://example.com",
                            "description": "after retry",
                        }
                    ]
                }
            }

            class SuccessResponse:
                def raise_for_status(self):
                    return None

                def json(self):
                    return success_payload

            class MockClient:
                def __init__(self):
                    self.calls = 0

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def get(self, *args, **kwargs):
                    self.calls += 1
                    if self.calls == 1:
                        return RateLimitedResponse()
                    return SuccessResponse()

            client_instance = MockClient()

            with (
                patch("mcp_web_tools.search.httpx.AsyncClient", return_value=client_instance),
                patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            ):
                caplog.set_level(logging.INFO, logger="mcp_web_tools.search")
                result = await brave_search("test query", limit=2)

            assert result is not None
            assert result["provider"] == "brave"
            assert len(result["results"]) == 1
            assert client_instance.calls == 2
            mock_sleep.assert_awaited_once()
            assert mock_sleep.await_args_list == [call(1)]
            messages = [rec.getMessage() for rec in caplog.records]
            assert any("retrying in 1s" in message for message in messages)

    @pytest.mark.asyncio
    async def test_brave_search_logs_timeout_warning(self, caplog):
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test_key"}):
            class MockClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                async def get(self, *args, **kwargs):
                    raise httpx.TimeoutException("timeout")

            with patch("mcp_web_tools.search.httpx.AsyncClient", return_value=MockClient()):
                caplog.set_level(logging.WARNING, logger="mcp_web_tools.search")
                result = await brave_search("test query")

            assert result is None
            messages = [rec.getMessage() for rec in caplog.records]
            assert any("Brave Search API request timed out" in message for message in messages)


class TestPerplexitySearch:
    @pytest.mark.asyncio
    async def test_perplexity_search_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            result = await perplexity_search("test query")
            assert result is None

    @pytest.mark.asyncio
    async def test_perplexity_search_success(self):
        with (
            patch.dict(os.environ, {"PERPLEXITY_API_KEY": "perplex-key"}, clear=True),
            patch("mcp_web_tools.search.AsyncPerplexity") as mock_client_cls,
        ):
            mock_client = mock_client_cls.return_value
            mock_client.search.create = AsyncMock(
                return_value=SimpleNamespace(
                    results=[
                        Result(
                            title="Perplexity Title",
                            url="https://perplexity.ai",
                            snippet="Snippet",
                        )
                    ]
                )
            )

            result = await perplexity_search("test query", limit=5)

        mock_client_cls.assert_called_once_with(api_key="perplex-key")
        mock_client.search.create.assert_awaited_once_with(query="test query", max_results=5)
        assert result is not None
        assert result["provider"] == "perplexity"
        assert result["results"][0]["title"] == "Perplexity Title"
        assert result["results"][0]["url"] == "https://perplexity.ai"
        assert result["results"][0]["description"] == "Snippet"

    @pytest.mark.asyncio
    async def test_perplexity_search_handles_sdk_error(self, caplog):
        with (
            patch.dict(os.environ, {"PERPLEXITY_API_KEY": "perplex-key"}, clear=True),
            patch("mcp_web_tools.search.AsyncPerplexity") as mock_client_cls,
        ):
            mock_client = mock_client_cls.return_value
            mock_client.search.create = AsyncMock(side_effect=PerplexityError("boom"))

            caplog.set_level(logging.WARNING, logger="mcp_web_tools.search")
            result = await perplexity_search("test query")

        assert result is None
        assert any("Perplexity Search API failed" in rec.getMessage() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_perplexity_search_returns_none_on_empty_results(self):
        with (
            patch.dict(os.environ, {"PERPLEXITY_API_KEY": "perplex-key"}, clear=True),
            patch("mcp_web_tools.search.AsyncPerplexity") as mock_client_cls,
        ):
            mock_client = mock_client_cls.return_value
            mock_client.search.create = AsyncMock(return_value=SimpleNamespace(results=[]))

            result = await perplexity_search("test query")

        assert result is None


class TestGoogleSearch:
    def test_google_search_success(self):
        mock_result = Mock()
        mock_result.title = "Google Title"
        mock_result.url = "https://google.com"
        mock_result.description = "Google Description"
        
        with patch("googlesearch.search") as mock_search:
            mock_search.return_value = [mock_result]
            
            result = google_search("test query", limit=1)
            
            assert result is not None
            assert result["provider"] == "google"
            assert len(result["results"]) == 1
            assert result["results"][0]["title"] == "Google Title"
            assert result["results"][0]["url"] == "https://google.com"
            assert result["results"][0]["description"] == "Google Description"

    def test_google_search_no_results(self):
        with patch("googlesearch.search") as mock_search:
            mock_search.return_value = []
            
            result = google_search("test query")
            assert result is None

    def test_google_search_exception(self):
        with patch("googlesearch.search") as mock_search:
            mock_search.side_effect = Exception("Search failed")
            
            result = google_search("test query")
            assert result is None


class TestDuckDuckGoSearch:
    def test_duckduckgo_search_success(self):
        mock_results = [
            {
                "title": "DDG Title",
                "href": "https://ddg.com",
                "body": "DDG Body"
            }
        ]
        
        with patch("mcp_web_tools.search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = iter(mock_results)
            
            result = duckduckgo_search("test query", limit=1)
            
            assert result is not None
            assert result["provider"] == "duckduckgo"
            assert len(result["results"]) == 1
            assert result["results"][0]["title"] == "DDG Title"
            assert result["results"][0]["url"] == "https://ddg.com"
            assert result["results"][0]["description"] == "DDG Body"

    def test_duckduckgo_search_no_results(self):
        with patch("mcp_web_tools.search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = iter([])
            
            result = duckduckgo_search("test query")
            assert result is None

    def test_duckduckgo_search_exception(self):
        with patch("mcp_web_tools.search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.side_effect = Exception("Search failed")
            
            result = duckduckgo_search("test query")
            assert result is None


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_web_search_perplexity_success(self):
        perplexity_result = {
            "results": [{"title": "Perplexity", "url": "https://perplexity.ai", "description": "Perplexity desc"}],
            "provider": "perplexity",
        }

        with (
            patch("mcp_web_tools.search.perplexity_search", new_callable=AsyncMock) as mock_perplexity,
            patch("mcp_web_tools.search.brave_search", new_callable=AsyncMock) as mock_brave,
            patch("mcp_web_tools.search.google_search") as mock_google,
            patch("mcp_web_tools.search.duckduckgo_search") as mock_ddg,
        ):
            mock_perplexity.return_value = perplexity_result
            mock_brave.return_value = None
            mock_google.return_value = None
            mock_ddg.return_value = None

            result = await web_search("test query")

        assert result == perplexity_result
        mock_perplexity.assert_called_once_with("test query", 10)
        mock_brave.assert_not_called()
        mock_google.assert_not_called()
        mock_ddg.assert_not_called()

    @pytest.mark.asyncio
    async def test_web_search_brave_success(self):
        brave_result = {
            "results": [{"title": "Brave", "url": "https://brave.com", "description": "Brave desc"}],
            "provider": "brave"
        }

        with (
            patch("mcp_web_tools.search.perplexity_search", new_callable=AsyncMock) as mock_perplexity,
            patch("mcp_web_tools.search.brave_search", new_callable=AsyncMock) as mock_brave,
        ):
            mock_perplexity.return_value = None
            mock_brave.return_value = brave_result

            result = await web_search("test query")

        assert result == brave_result
        mock_perplexity.assert_called_once()
        mock_brave.assert_called_once_with("test query", 10)

    @pytest.mark.asyncio
    async def test_web_search_fallback_to_google(self):
        google_result = {
            "results": [{"title": "Google", "url": "https://google.com", "description": "Google desc"}],
            "provider": "google"
        }
        
        with (
            patch("mcp_web_tools.search.perplexity_search", new_callable=AsyncMock) as mock_perplexity,
            patch("mcp_web_tools.search.brave_search", new_callable=AsyncMock) as mock_brave,
            patch("mcp_web_tools.search.google_search") as mock_google,
        ):
            mock_perplexity.return_value = None
            mock_brave.return_value = None
            mock_google.return_value = google_result

            result = await web_search("test query")

        assert result == google_result
        mock_perplexity.assert_called_once()
        mock_brave.assert_called_once()
        mock_google.assert_called_once()

    @pytest.mark.asyncio
    async def test_web_search_fallback_to_duckduckgo(self):
        ddg_result = {
            "results": [{"title": "DDG", "url": "https://ddg.com", "description": "DDG desc"}],
            "provider": "duckduckgo"
        }
        
        with (
            patch("mcp_web_tools.search.perplexity_search", new_callable=AsyncMock) as mock_perplexity,
            patch("mcp_web_tools.search.brave_search", new_callable=AsyncMock) as mock_brave,
            patch("mcp_web_tools.search.google_search") as mock_google,
            patch("mcp_web_tools.search.duckduckgo_search") as mock_ddg,
        ):
            mock_perplexity.return_value = None
            mock_brave.return_value = None
            mock_google.return_value = None
            mock_ddg.return_value = ddg_result

            result = await web_search("test query")

        assert result == ddg_result
        mock_perplexity.assert_called_once()
        mock_brave.assert_called_once()
        mock_google.assert_called_once()
        mock_ddg.assert_called_once()

    @pytest.mark.asyncio
    async def test_web_search_all_fail(self):
        with (
            patch("mcp_web_tools.search.perplexity_search", new_callable=AsyncMock) as mock_perplexity,
            patch("mcp_web_tools.search.brave_search", new_callable=AsyncMock) as mock_brave,
            patch("mcp_web_tools.search.google_search") as mock_google,
            patch("mcp_web_tools.search.duckduckgo_search") as mock_ddg,
        ):
            mock_perplexity.return_value = None
            mock_brave.return_value = None
            mock_google.return_value = None
            mock_ddg.return_value = None

            result = await web_search("test query")

        assert result == {"results": [], "provider": "none"}
        mock_perplexity.assert_called_once()
        mock_brave.assert_called_once()
        mock_google.assert_called_once()
        mock_ddg.assert_called_once()
