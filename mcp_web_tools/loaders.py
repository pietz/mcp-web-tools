import os
import io
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
            # Initialize html and browser to None
            html = None
            browser = None

            try:
                logger.info(f"Attempting to fetch {url} with trafilatura")
                html = trafilatura.fetch_url(url)
            except Exception as e:
                logger.error(f"Error fetching page with trafilatura: {str(e)}")

            if not html:
                try:
                    browser = await zd.start(headless=True, sandbox=False)
                    page = await browser.get(url)
                    await page.wait_for_ready_state("complete", timeout=5)
                    html = await page.get_content()
                except Exception as e:
                    logger.warning(
                        f"Error fetching page with zendriver: {str(e)}, trying trafilatura next"
                    )
                finally:
                    # Ensure browser is closed even if an error occurs
                    if browser:
                        try:
                            await browser.stop()
                        except Exception:
                            pass  # Ignore errors during browser closing

            # If both methods failed, return error
            if not html:
                logger.error(
                    f"Failed to retrieve content from {url} using both zendriver and trafilatura"
                )
                return f"Error: Failed to retrieve page content from {url} using multiple methods"

            if raw:
                res = html[offset : offset + limit]
                res += f"\n\n---Showing {offset} to {min(offset + limit, len(html))} out of {len(html)} characters.---"
                return res

            try:
                content = trafilatura.extract(
                    html,
                    output_format="markdown",
                    include_images=True,
                    include_links=True,
                )
            except Exception as e:
                logger.error(f"Error extracting content with trafilatura: {str(e)}")
                return f"Error: Failed to extract readable content: {str(e)}"

            if not content:
                logger.warning(f"Failed to extract content from {url}")
                # Fallback to raw HTML with a warning
                return f"Warning: Could not extract readable content from {url}. Showing raw HTML instead.\n\n{html[offset : offset + limit]}"

            res = content[offset : offset + limit]
            res += f"\n\n---Showing {offset} to {min(offset + limit, len(content))} out of {len(content)} characters.---"
            return res

    except asyncio.TimeoutError:
        logger.error(f"Request timed out after 10 seconds for URL: {url}")
        return f"Error: Request timed out after 10 seconds for URL: {url}"
    except Exception as e:
        logger.error(f"Error loading page: {str(e)}")
        return f"Error loading page: {str(e)}"


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
