import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mcp_web_tools.loaders import (
    capture_webpage_screenshot,
    load_content,
    load_webpage,
)


@pytest.mark.asyncio
async def test_load_webpage_uses_trafilatura_first():
    html = "<html><body>Hello</body></html>"
    markdown = "Hello from page"

    with (
        patch("mcp_web_tools.loaders.trafilatura.fetch_url", return_value=html) as mock_fetch,
        patch("mcp_web_tools.loaders.trafilatura.extract", return_value=markdown) as mock_extract,
        patch("mcp_web_tools.loaders.zd.start", new_callable=AsyncMock) as mock_start,
    ):
        result = await load_webpage("https://example.com/page", limit=100)

    assert result.startswith("_Fetched via: trafilatura_")
    assert "Hello from page" in result
    assert "---Showing 0 to" in result
    mock_fetch.assert_called_once_with("https://example.com/page")
    mock_extract.assert_called_once()
    mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_load_webpage_falls_back_to_zendriver():
    html = "<html>From Zendriver</html>"
    markdown = "Zendriver content"

    page = SimpleNamespace(
        wait_for_ready_state=AsyncMock(return_value=None),
        wait=AsyncMock(return_value=None),
        get_content=AsyncMock(return_value=html),
    )
    browser = SimpleNamespace(
        get=AsyncMock(return_value=page),
        stop=AsyncMock(return_value=None),
    )

    with (
        patch("mcp_web_tools.loaders.trafilatura.fetch_url", return_value=None),
        patch("mcp_web_tools.loaders.zd.start", new_callable=AsyncMock) as mock_start,
        patch("mcp_web_tools.loaders.trafilatura.extract", return_value=markdown),
    ):
        mock_start.return_value = browser
        result = await load_webpage("https://example.com/page")

    assert result.startswith("_Fetched via: zendriver_")
    assert "Zendriver content" in result
    assert browser.get.await_args_list[0].args == ("https://example.com/page",)
    page.wait_for_ready_state.assert_awaited_once_with("complete", timeout=5)
    page.wait.assert_awaited_once_with(t=1)
    page.get_content.assert_awaited_once_with()
    browser.stop.assert_awaited_once_with()
    mock_start.assert_awaited_once_with(headless=True, sandbox=False)


@pytest.mark.asyncio
async def test_load_webpage_raw_html_short_circuits_extraction():
    html = "<html>Example</html>"

    with (
        patch("mcp_web_tools.loaders.trafilatura.fetch_url", return_value=html),
        patch("mcp_web_tools.loaders.trafilatura.extract") as mock_extract,
        patch("mcp_web_tools.loaders.zd.start", new_callable=AsyncMock) as mock_start,
    ):
        result = await load_webpage("https://example.com/raw", limit=5, raw=True)

    assert result.startswith("_Fetched via: trafilatura_")
    assert "<html" in result
    assert mock_extract.call_count == 0
    mock_start.assert_not_called()


@pytest.mark.asyncio
async def test_load_webpage_returns_warning_when_extraction_empty():
    html = "<html><body>No extract</body></html>"

    with (
        patch("mcp_web_tools.loaders.trafilatura.fetch_url", return_value=html),
        patch("mcp_web_tools.loaders.trafilatura.extract", return_value=""),
    ):
        result = await load_webpage("https://example.com/empty")

    assert "Warning: Could not extract readable content" in result
    assert "<html><body>No extract</body></html>" in result


@pytest.mark.asyncio
async def test_load_webpage_timeout_error():
    class TimeoutContext:
        async def __aenter__(self):
            raise asyncio.TimeoutError

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with patch("mcp_web_tools.loaders.asyncio.timeout", return_value=TimeoutContext()):
        result = await load_webpage("https://example.com/timeout")

    assert result == "Error: Request timed out after 10 seconds for URL: https://example.com/timeout"


@pytest.mark.asyncio
async def test_load_content_dispatches_to_image_loader():
    with (
        patch("mcp_web_tools.loaders.load_image_file", new_callable=AsyncMock) as mock_image,
        patch("mcp_web_tools.loaders.load_pdf_document", new_callable=AsyncMock) as mock_pdf,
        patch("mcp_web_tools.loaders.load_webpage", new_callable=AsyncMock) as mock_webpage,
    ):
        mock_image.return_value = "image"
        result = await load_content("https://example.com/picture.PNG")

    assert result == "image"
    mock_image.assert_awaited_once_with("https://example.com/picture.PNG")
    mock_pdf.assert_not_called()
    mock_webpage.assert_not_called()


@pytest.mark.asyncio
async def test_load_content_dispatches_to_pdf_loader():
    with (
        patch("mcp_web_tools.loaders.load_image_file", new_callable=AsyncMock) as mock_image,
        patch("mcp_web_tools.loaders.load_pdf_document", new_callable=AsyncMock) as mock_pdf,
        patch("mcp_web_tools.loaders.load_webpage", new_callable=AsyncMock) as mock_webpage,
    ):
        mock_pdf.return_value = "pdf"
        result = await load_content("https://example.com/manual.pdf", limit=500, offset=10, raw=True)

    assert result == "pdf"
    mock_pdf.assert_awaited_once_with("https://example.com/manual.pdf", 500, 10, True)
    mock_image.assert_not_called()
    mock_webpage.assert_not_called()


@pytest.mark.asyncio
async def test_load_content_dispatches_to_webpage_loader():
    with (
        patch("mcp_web_tools.loaders.load_image_file", new_callable=AsyncMock) as mock_image,
        patch("mcp_web_tools.loaders.load_pdf_document", new_callable=AsyncMock) as mock_pdf,
        patch("mcp_web_tools.loaders.load_webpage", new_callable=AsyncMock) as mock_webpage,
    ):
        mock_webpage.return_value = "page"
        result = await load_content("https://example.com/article", limit=750, offset=5, raw=False)

    assert result == "page"
    mock_webpage.assert_awaited_once_with("https://example.com/article", 750, 5, False)
    mock_image.assert_not_called()
    mock_pdf.assert_not_called()


@pytest.mark.asyncio
async def test_load_content_propagates_image_value_error():
    with patch(
        "mcp_web_tools.loaders.load_image_file",
        new_callable=AsyncMock,
        side_effect=ValueError("bad image"),
    ) as mock_image:
        with pytest.raises(ValueError):
            await load_content("https://example.com/fail.jpg")

    mock_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_content_wraps_other_exceptions():
    with (
        patch("mcp_web_tools.loaders.load_image_file", new_callable=AsyncMock) as mock_image,
        patch("mcp_web_tools.loaders.load_pdf_document", new_callable=AsyncMock) as mock_pdf,
        patch(
            "mcp_web_tools.loaders.load_webpage",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ) as mock_webpage,
    ):
        result = await load_content("https://example.com/article")

    assert result == "Error loading content from https://example.com/article: boom"
    mock_webpage.assert_awaited_once_with("https://example.com/article", 10_000, 0, False)
    mock_image.assert_not_called()
    mock_pdf.assert_not_called()


@pytest.mark.asyncio
async def test_capture_webpage_screenshot_returns_image():
    screenshot_bytes = b"fakepng"
    screenshot_b64 = base64.b64encode(screenshot_bytes).decode()

    tab = SimpleNamespace(
        wait_for_ready_state=AsyncMock(return_value=None),
        wait=AsyncMock(return_value=None),
        screenshot_b64=AsyncMock(return_value=screenshot_b64),
    )
    browser = SimpleNamespace(
        get=AsyncMock(return_value=tab),
        stop=AsyncMock(return_value=None),
    )

    with patch("mcp_web_tools.loaders.zd.start", new_callable=AsyncMock) as mock_start:
        mock_start.return_value = browser
        result = await capture_webpage_screenshot(
            "https://example.com", full_page=False
        )

    assert result.data == screenshot_bytes
    assert result._format == "png"
    browser.get.assert_awaited_once_with("https://example.com")
    tab.wait_for_ready_state.assert_awaited_once_with("complete", timeout=5)
    tab.wait.assert_awaited_once_with(t=1)
    tab.screenshot_b64.assert_awaited_once_with(format="png", full_page=False)
    browser.stop.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_capture_webpage_screenshot_raises_when_no_data():
    tab = SimpleNamespace(
        wait_for_ready_state=AsyncMock(return_value=None),
        wait=AsyncMock(return_value=None),
        screenshot_b64=AsyncMock(return_value=""),
    )
    browser = SimpleNamespace(
        get=AsyncMock(return_value=tab),
        stop=AsyncMock(return_value=None),
    )

    with patch("mcp_web_tools.loaders.zd.start", new_callable=AsyncMock) as mock_start:
        mock_start.return_value = browser
        with pytest.raises(ValueError, match="No screenshot data returned"):
            await capture_webpage_screenshot("https://example.com/missing")

    browser.stop.assert_awaited_once_with()
    tab.wait.assert_awaited_once_with(t=1)
    tab.screenshot_b64.assert_awaited_once_with(format="png", full_page=False)
