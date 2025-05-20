import os
import asyncio
from mcp.server.fastmcp import FastMCP, Image
from pydantic import Field
import logging
import io
from PIL import Image as PILImage

from duckduckgo_search import DDGS
import googlesearch

import httpx
import pymupdf
import zendriver as zd  # fetching
import trafilatura  # web extraction
import pymupdf4llm  # pdf extraction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Web Tools", log_level="INFO")


@mcp.prompt()
def help() -> str:
    """Load detailed information about the server and its usage."""
    return """
    ## Summary
    This server provides tools for web searching and content extraction.

    ## Usage
    1. Use `web_search` to find potentially relevant URLs based on your query.
    2. Use `load_page` or `load_pdf` to fetch and extract URLs of interest.
    3. Use `load_image` to fetch and display images from the web.

    ## Notes
    - Rely on unbiased and trusted sources to retrieve accurate results.
    - Use `raw` only if the Markdown extraction fails or to inspect the raw HTML.
    - Images are automatically resized to fit within 1024x1024 dimensions.
    """


@mcp.tool()
async def web_search(
    query: str = Field(description="The search query to use."),
    limit: int = Field(10, le=30, description="Max. number of results to return."),
    offset: int = Field(0, ge=0, description="Result offset to start returning from."),
) -> list[dict]:
    """
    Execute a web search using the given search query.
    Tries to use Brave first, then Google, finally DuckDuckGo as fallbacks.
    Returns a list of the title, URL, and description of each result.
    """

    if os.getenv("BRAVE_SEARCH_API_KEY"):
        try:
            logger.info("Using Brave Search...")

            async with httpx.AsyncClient() as client:
                url = "https://api.search.brave.com/res/v1/web/search"
                headers = {"X-Subscription-Token": os.getenv("BRAVE_SEARCH_API_KEY")}
                params = {"q": query, "count": limit}
                r = await client.get(url, headers=headers, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                if "web" not in data or "results" not in data["web"]:
                    raise ValueError("Unexpected response format from Brave Search API")
                results = data["web"]["results"]
                return [
                    {
                        "title": x["title"],
                        "url": x["url"],
                        "description": x["description"],
                    }
                    for x in results
                ]
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"Brave Search API returned status code {e.response.status_code}"
            )
        except httpx.TimeoutException:
            logger.warning("Brave Search API request timed out")
        except Exception as e:
            logger.warning(f"Error using Brave Search: {str(e)}")

    try:
        logger.info("Using Google Search...")
        results = googlesearch.search(query, num_results=limit, advanced=True)
        if not results:
            raise ValueError("No results returned from Google Search")
        return [
            {"title": r.title, "url": r.url, "description": r.description}
            for r in results
        ]
    except Exception as e:
        logger.warning(f"Error using Google Search: {str(e)}")

    try:
        logger.info("Using DuckDuckGo Search...")
        results = list(DDGS().text(query, max_results=limit))
        if not results:
            raise ValueError("No results returned from DuckDuckGo")
        return [
            {"title": r["title"], "url": r["href"], "description": r["body"]}
            for r in results
        ]
    except Exception as e:
        logger.warning(f"Error using DuckDuckGo: {str(e)}")

    logger.error("All search methods failed.")
    return []  # Return empty list instead of error string to maintain return type


@mcp.tool()
async def load_page(
    url: str = Field(description="The remote URL to load/fetch content from."),
    limit: int = Field(
        10_000, gt=0, le=100_000, description="Max. number of characters to return."
    ),
    offset: int = Field(
        0, ge=0, description="Character offset to start returning from."
    ),
    raw: bool = Field(
        False, description="Return raw HTML instead of cleaned Markdown."
    ),
) -> str:
    """
    Fetch the content from an URL and return it in cleaned Markdown format.
    Use `offset` if you need to paginate/scroll the content.
    Use `raw` to retrieve the original source code without trying to clean it.
    """

    try:
        async with asyncio.timeout(10):
            try:
                browser = await zd.start(headless=True, sandbox=False)
                page = await browser.get(url)
                await page.wait_for_ready_state("complete", timeout=5)
                html = await page.get_content()
            except Exception as e:
                logger.error(f"Error fetching page with zendriver: {str(e)}")
                return f"Error: Failed to retrieve page content: {str(e)}"
            finally:
                try:
                    await browser.stop()
                except Exception:
                    pass  # Ignore errors during browser closing

            if not html:
                logger.error(f"Received empty HTML from {url}")
                return f"Error: Retrieved empty content from {url}"

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


@mcp.tool()
async def load_pdf(
    url: str = Field(description="The remote PDF file URL to fetch."),
    limit: int = Field(
        10_000, gt=0, le=100_000, description="Max. number of characters to return."
    ),
    offset: int = Field(0, ge=0, description="Starting index of the content"),
    raw: bool = Field(
        False, description="Return raw content instead of cleaned Markdown."
    ),
) -> str:
    """
    Fetch a PDF file from the internet and extract its content in markdown.
    Use `offset` if you need to paginate/scroll the content.
    Use `raw` to retrieve the original source code without trying to format it.
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


@mcp.tool()
async def load_image(
    url: str = Field(description="The remote image file URL to fetch."),
) -> Image:
    """
    Fetch an image from the internet and view it.
    """
    try:
        async with asyncio.timeout(10):
            res = httpx.get(url, follow_redirects=True)
            if res.status_code != 200:
                logger.error(f"Failed to fetch image from {url}")
                raise ValueError(f"Error: Could not fetch image from {url}")

            img = PILImage.open(io.BytesIO(res.content))
            if max(img.size) > 1600:
                img.thumbnail((1600, 1600))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return Image(data=buffer.getvalue(), format="png")

    except asyncio.TimeoutError:
        logger.error(f"Request timed out after 10 seconds for URL: {url}")
        raise ValueError(f"Error: Request timed out after 10 seconds for URL: {url}")
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise ValueError(f"Error loading image: {str(e)}")


def main():
    mcp.run()


if __name__ == "__main__":
    main()
