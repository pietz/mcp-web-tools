import os
import io
import base64
import logging
import asyncio

import httpx
import pymupdf
from PIL import Image as PILImage
import zendriver as zd  # fetching
import trafilatura  # web extraction
import pymupdf4llm  # pdf extraction

from mcp.server.fastmcp import Image

logger = logging.getLogger(__name__)


def _fetch_with_trafilatura(url: str) -> tuple[str | None, str | None]:
    try:
        logger.info("Attempting to fetch %s with trafilatura", url)
        html = trafilatura.fetch_url(url)
        if html:
            return html, "trafilatura"
    except Exception as exc:
        logger.error("Error fetching page with trafilatura: %s", exc)
    return None, None


async def _fetch_with_zendriver(url: str) -> tuple[str | None, str | None]:
    browser = None
    try:
        browser = await zd.start(headless=True, sandbox=False)
        page = await browser.get(url)
        await page.wait_for_ready_state("complete", timeout=5)
        await page.wait(t=1)  # Allow dynamic content to settle
        html = await page.get_content()
        if html:
            return html, "zendriver"
    except Exception as exc:
        logger.warning("Error fetching page with zendriver: %s", exc)
    finally:
        if browser:
            try:
                await browser.stop()
            except Exception:
                pass
    return None, None


async def _fetch_html(url: str) -> tuple[str | None, str | None]:
    html, provider = _fetch_with_trafilatura(url)
    if html:
        return html, provider
    return await _fetch_with_zendriver(url)


def _extract_markdown(html: str) -> tuple[str | None, str | None]:
    try:
        content = trafilatura.extract(
            html,
            output_format="markdown",
            include_images=True,
            include_links=True,
        )
    except Exception as exc:
        logger.error("Error extracting content with trafilatura: %s", exc)
        return None, f"Error: Failed to extract readable content: {exc}"
    if not content:
        return None, None
    return content, None


def _format_frontmatter(
    *,
    fetched: str | None,
    extracted: str | None,
    start: int,
    end: int,
    length: int,
) -> str:
    lines = ["---"]
    lines.append(f"fetched: {fetched or 'unknown'}")
    lines.append(f"extracted: {extracted or 'none'}")
    lines.append(f"start: {start}")
    lines.append(f"end: {end}")
    lines.append(f"length: {length}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _slice_text(text: str, start: int, limit: int) -> tuple[str, int]:
    if limit <= 0:
        end = len(text)
    else:
        end = min(start + limit, len(text))
    return text[start:end], end


async def load_webpage(
    url: str, limit: int = 10_000, offset: int = 0, raw: bool = False
) -> str:
    """
    Fetch the content from a URL and return it in cleaned Markdown format.
    Args:
        url: The URL to fetch content from
        limit: Maximum number of characters to return
        offset: Character offset to start from
        raw: If True, returns raw HTML instead of cleaned Markdown
    Returns:
        Extracted content as string (Markdown or raw HTML)
    """
    try:
        async with asyncio.timeout(10):
            html, fetch_provider = await _fetch_html(url)
            if not html:
                logger.error(
                    "Failed to retrieve content from %s using both zendriver and trafilatura",
                    url,
                )
                return f"Error: Failed to retrieve page content from {url} using multiple methods"

            extraction_provider: str | None = None
            warning: str | None = None
            source = html

            if raw:
                extraction_provider = "raw"
            else:
                content, extraction_error = _extract_markdown(html)
                if extraction_error:
                    return extraction_error
                if content:
                    source = content
                    extraction_provider = "trafilatura"
                else:
                    extraction_provider = "raw"
                    warning = (
                        f"Warning: Could not extract readable content from {url}. "
                        "Showing raw HTML instead."
                    )

            total_length = len(source)
            content_slice, slice_end = _slice_text(source, offset, limit)
            frontmatter = _format_frontmatter(
                fetched=fetch_provider,
                extracted=extraction_provider,
                start=offset,
                end=slice_end,
                length=total_length,
            )

            parts = [frontmatter]
            if warning:
                parts.append(f"{warning}\n\n")
            parts.append(content_slice)
            return "".join(parts)

    except asyncio.TimeoutError:
        logger.error("Request timed out after 10 seconds for URL: %s", url)
        return f"Error: Request timed out after 10 seconds for URL: {url}"
    except Exception as e:
        logger.error("Error loading page: %s", e)
        return f"Error loading page: {str(e)}"


async def capture_webpage_screenshot(url: str, *, full_page: bool = False) -> Image:
    """Render a webpage in a headless browser and return a PNG screenshot.

    Args:
        url: The URL of the page to capture.
        full_page: When True capture the entire scrollable page; defaults to the visible viewport only.

    Returns:
        An :class:`~mcp.server.fastmcp.Image` containing PNG image bytes.

    Raises:
        ValueError: If the screenshot cannot be captured.
    """

    browser = None
    try:
        async with asyncio.timeout(10):
            browser = await zd.start(headless=True, sandbox=False)
            tab = await browser.get(url)
            await tab.wait_for_ready_state("complete", timeout=5)
            await tab.wait(t=1)
            screenshot_b64 = await tab.screenshot_b64(format="png", full_page=full_page)

            if not screenshot_b64:
                raise RuntimeError("No screenshot data returned from zendriver")

            try:
                screenshot_bytes = base64.b64decode(screenshot_b64)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise RuntimeError(f"Invalid screenshot data: {exc}") from exc

            return Image(data=screenshot_bytes, format="png")

    except asyncio.TimeoutError:
        logger.error(f"Screenshot request timed out after 10 seconds for URL: {url}")
        raise ValueError(
            f"Error: Screenshot request timed out after 10 seconds for URL: {url}"
        )
    except Exception as e:
        logger.error(f"Error capturing screenshot for {url}: {str(e)}")
        raise ValueError(f"Error capturing screenshot for {url}: {str(e)}") from e
    finally:
        if browser:
            try:
                await browser.stop()
            except Exception:
                pass


async def load_pdf_document(
    url: str, limit: int = 10_000, offset: int = 0, raw: bool = False
) -> str:
    """
    Fetch a PDF file from the internet and extract its content.
    Args:
        url: URL to the PDF file
        limit: Maximum number of characters to return
        offset: Character offset to start from
        raw: If True, returns raw text instead of formatted Markdown
    Returns:
        Extracted content as string
    """
    try:
        async with asyncio.timeout(15):  # Allow more time for PDFs which can be large
            res = httpx.get(url, follow_redirects=True, timeout=10)
            res.raise_for_status()

            try:
                doc = pymupdf.Document(stream=res.content)
                if raw:
                    pages = [page.get_text() for page in doc]
                    content = "\n---\n".join(pages)
                else:
                    content = pymupdf4llm.to_markdown(doc)
                doc.close()

                if not content or content.strip() == "":
                    logger.warning(f"Extracted empty content from PDF at {url}")
                    return f"Warning: PDF was retrieved but no text content could be extracted from {url}"

                result = content[offset : offset + limit]
                if len(content) > offset + limit:
                    result += f"\n\n---Showing {offset} to {min(offset + limit, len(content))} out of {len(content)} characters.---"
                return result

            except Exception as e:
                logger.error(f"Error processing PDF content: {str(e)}")
                return f"Error: PDF was downloaded but could not be processed: {str(e)}"

    except asyncio.TimeoutError:
        logger.error(f"Request timed out after 15 seconds for PDF URL: {url}")
        return f"Error: Request timed out after 15 seconds for PDF URL: {url}"
    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error {e.response.status_code} when fetching PDF from {url}"
        )
        return (
            f"Error: HTTP status {e.response.status_code} when fetching PDF from {url}"
        )
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        return f"Error loading PDF: {str(e)}"


async def load_image_file(url: str) -> Image:
    """
    Fetch an image from the internet and return it as a processed Image object.
    Args:
        url: URL to the image file
    Returns:
        Image object with processed image data
    Raises:
        ValueError: If image cannot be fetched or processed
    """
    try:
        async with asyncio.timeout(10):
            res = httpx.get(url, follow_redirects=True)
            if res.status_code != 200:
                logger.error(f"Failed to fetch image from {url}")
                raise ValueError(f"Error: Could not fetch image from {url}")

            img = PILImage.open(io.BytesIO(res.content))
            if max(img.size) > 1536:
                img.thumbnail((1536, 1536))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return Image(data=buffer.getvalue(), format="png")

    except asyncio.TimeoutError:
        logger.error(f"Request timed out after 10 seconds for URL: {url}")
        raise ValueError(f"Error: Request timed out after 10 seconds for URL: {url}")
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise ValueError(f"Error loading image: {str(e)}")


async def load_content(
    url: str, limit: int = 10_000, offset: int = 0, raw: bool = False
):
    """
    Universal content loader that handles different content types based on URL pattern.

    Args:
        url: The URL to fetch content from
        limit: Maximum number of characters to return (for text content)
        offset: Character offset to start from (for text content)
        raw: If True, returns raw content instead of processed format

    Returns:
        Extracted content as string or Image object depending on content type

    Raises:
        ValueError: If image loading fails
    """
    # Check URL pattern to guess content type
    url_lower = url.lower()

    # Extract extension if present
    path = url.split("?")[0].split("#")[0]  # Remove query params and fragments
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"):
        content_type = "image"
    elif ext == ".pdf" or url_lower.endswith("/pdf") or "pdf" in url_lower:
        content_type = "pdf"
    else:
        # Default to webpage
        content_type = "webpage"

    logger.info(f"Auto-detected content type '{content_type}' for URL: {url}")

    # Load content based on detected type
    try:
        if content_type == "image":
            return await load_image_file(url)
        elif content_type == "pdf":
            return await load_pdf_document(url, limit, offset, raw)
        else:  # webpage
            return await load_webpage(url, limit, offset, raw)
    except ValueError as e:
        # Re-raise ValueError from image loader
        raise e
    except Exception as e:
        logger.error(f"Error in universal content loader: {str(e)}")
        return f"Error loading content from {url}: {str(e)}"
