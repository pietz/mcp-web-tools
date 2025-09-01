import pytest
import os
from unittest.mock import patch, AsyncMock, Mock
import httpx

from mcp_web_tools.search import brave_search, google_search, duckduckgo_search, web_search


class TestBraveSearch:
    @pytest.mark.asyncio
    async def test_brave_search_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            result = await brave_search("test query")
            assert result is None

    @pytest.mark.asyncio
    async def test_brave_search_success(self):
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test_key"}):
            from types import SimpleNamespace
            with patch("mcp_web_tools.search.BraveSearch") as mock_bs:
                instance = mock_bs.return_value
                result_obj = SimpleNamespace(
                    title="Test Title", url="https://test.com", description="Test Description"
                )
                mock_resp = SimpleNamespace(web=SimpleNamespace(results=[result_obj]))
                instance.web = AsyncMock(return_value=mock_resp)

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
            with patch("mcp_web_tools.search.BraveSearch") as mock_bs:
                instance = mock_bs.return_value
                instance.web = AsyncMock(side_effect=Exception("API error"))

                result = await brave_search("test query")
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
    async def test_web_search_brave_success(self):
        brave_result = {
            "results": [{"title": "Brave", "url": "https://brave.com", "description": "Brave desc"}],
            "provider": "brave"
        }
        
        with patch("mcp_web_tools.search.brave_search", new_callable=AsyncMock) as mock_brave:
            mock_brave.return_value = brave_result
            
            result = await web_search("test query")
            
            assert result == brave_result
            mock_brave.assert_called_once_with("test query", 10)

    @pytest.mark.asyncio
    async def test_web_search_fallback_to_google(self):
        google_result = {
            "results": [{"title": "Google", "url": "https://google.com", "description": "Google desc"}],
            "provider": "google"
        }
        
        with patch("mcp_web_tools.search.brave_search", new_callable=AsyncMock) as mock_brave:
            with patch("mcp_web_tools.search.google_search") as mock_google:
                mock_brave.return_value = None
                mock_google.return_value = google_result
                
                result = await web_search("test query")
                
                assert result == google_result
                mock_brave.assert_called_once()
                mock_google.assert_called_once()

    @pytest.mark.asyncio
    async def test_web_search_fallback_to_duckduckgo(self):
        ddg_result = {
            "results": [{"title": "DDG", "url": "https://ddg.com", "description": "DDG desc"}],
            "provider": "duckduckgo"
        }
        
        with patch("mcp_web_tools.search.brave_search", new_callable=AsyncMock) as mock_brave:
            with patch("mcp_web_tools.search.google_search") as mock_google:
                with patch("mcp_web_tools.search.duckduckgo_search") as mock_ddg:
                    mock_brave.return_value = None
                    mock_google.return_value = None
                    mock_ddg.return_value = ddg_result
                    
                    result = await web_search("test query")
                    
                    assert result == ddg_result
                    mock_brave.assert_called_once()
                    mock_google.assert_called_once()
                    mock_ddg.assert_called_once()

    @pytest.mark.asyncio
    async def test_web_search_all_fail(self):
        with patch("mcp_web_tools.search.brave_search", new_callable=AsyncMock) as mock_brave:
            with patch("mcp_web_tools.search.google_search") as mock_google:
                with patch("mcp_web_tools.search.duckduckgo_search") as mock_ddg:
                    mock_brave.return_value = None
                    mock_google.return_value = None
                    mock_ddg.return_value = None
                    
                    result = await web_search("test query")
                    
                    assert result == {"results": [], "provider": "none"}
                    mock_brave.assert_called_once()
                    mock_google.assert_called_once()
                    mock_ddg.assert_called_once()
