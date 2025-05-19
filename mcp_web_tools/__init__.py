import os
import asyncio
from mcp.server.fastmcp import FastMCP
from pydantic import Field
import logging

from brave_search_python_client import BraveSearch, WebSearchRequest
from duckduckgo_search import DDGS
import googlesearch

import httpx
import pymupdf
import zendriver  # fetching
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

    ## Notes
    - Rely on unbiased and trusted sources to retrieve accurate results.
    - Use `raw` only if the Markdown extraction fails or to inspect the raw HTML.
    """


@mcp.tool()
async def web_search(
    query: str = Field(description="The search query to use."),
    limit: int = Field(10, le=30, description="Max. number of results to return."),
    offset: int = Field(0, ge=0, description="Result offset to start returning from."),
) -> list[dict]:
    """
    Execute a web search using the given search query.
    Searches Brave first, then Google, finally DuckDuckGo.
    Returns a list of the title, URL, and description of each result.
    """

    if os.getenv("BRAVE_SEARCH_API_KEY"):
        try:
            logger.info("Using Brave Search...")
            brave = BraveSearch(os.getenv("BRAVE_SEARCH_API_KEY"))
            res = await brave.web(WebSearchRequest(q=query, count=limit))
            return [
                {"title": x.title, "url": x.url, "description": x.description}
                for x in res.web.results
            ]
        except Exception as e:
            logger.warning("Error using Brave Search:")
            logger.warning(e)

    try:
        logger.info("Using Google Search...")
        results = googlesearch.search(query, num_results=limit, advanced=True)
        return [
            {"title": r.title, "url": r.url, "description": r.description}
            for r in results
        ]
    except Exception as e:
        logger.warning("Error using Google Search:")
        logger.warning(e)

    try:
        logger.info("Using DuckDuckGo Search...")
        results = DDGS().text(query, max_results=limit)
        return [
            {"title": r["title"], "url": r["href"], "description": r["body"]}
            for r in results
        ]
    except Exception as e:
        logger.warning("Error using DuckDuckGo:")
        logger.warning(e)

    logger.error("All search methods failed.")
    return "Error: Could not fetch search results."


@mcp.tool()
async def load_page(
    url: str = Field(description="The remote URL to load/fetch content from."),
    limit: int = Field(
        10_000, gt=0, le=100_000, description="Max. number of characters to return."
    ),
    offset: int = Field(
        0, ge=0, description="Charecter offset to start returning from."
    ),
    raw: bool = Field(
        False, description="Return raw HTML instead of cleaned Markdown."
    ),
) -> str:
    """Fetch the content from an URL and return it in cleaned Markdown format."""

    try:
        async with asyncio.timeout(10):
            html = trafilatura.fetch_url(url)

            if not html:
                logger.warning(f"Trafilatura failed for {url}")
                browser = await zendriver.start(headless=True)
                page = await browser.get(url)
                await page.wait_for_ready_state("complete")
                await asyncio.sleep(1) # Increases success rate
                html = await page.get_content()
                await browser.stop()

            if raw:
                return html[offset : offset + limit]

            content = trafilatura.extract(
                html,
                output_format="markdown",
                include_images=True,
                include_links=True,
            )

            if not content:
                logger.error(f"Failed to extract content from {url}")
                return f"Error: Could not extract content from {url}"

            return content[offset : offset + limit]

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
    """Fetch a PDF file from the internet and extract its content in markdown."""
    res = httpx.get(url, follow_redirects=True)
    if res.status_code != 200:
        logger.error(f"Failed to fetch PDF from {url}")
        return f"Error: Could not fetch PDF from {url}"
    
    doc = pymupdf.Document(stream=res.content)
    if raw:
        pages = [page.get_text() for page in doc]
        content = "\n---\n".join(pages)
    else:
        content = pymupdf4llm.to_markdown(doc)
    doc.close()
    return content[offset : offset + limit]


def main():
    """Entry point for the package when installed via pip."""
    mcp.run()

if __name__ == "__main__":
    main()
