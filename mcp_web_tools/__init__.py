from typing import Annotated

from pydantic import Field

from mcp.server import FastMCP
from mcp.server.fastmcp import Image

from .search import web_search
from .loaders import load_content, capture_webpage_screenshot

# Create the MCP server
mcp = FastMCP("Web Tools", log_level="INFO")


@mcp.tool()
async def search_web(
    query: Annotated[str, Field(description="The search query to use.")],
    offset: Annotated[
        int,
        Field(
            ge=0,
            description="To scroll through more results.",
        ),
    ] = 0,
) -> dict:
    """
    Execute a web search using the given search query and returns 10 results.
    Tries to use Brave first, then Google, finally DuckDuckGo as fallbacks.
    Returns a dictionary with 'results' (list of search results) and 'provider' (search engine used).
    """
    return await web_search(query, 10, offset)


@mcp.tool()
async def fetch_url(
    url: Annotated[str, Field(description="The remote URL to load content from.")],
    offset: Annotated[
        int,
        Field(
            ge=0,
            description="Character/content offset to start from (for text content).",
        ),
    ] = 0,
    raw: Annotated[
        bool,
        Field(
            description="Return raw content instead cleaning it. Relevant for webpages and PDFs.",
        ),
    ] = False,
):
    """
    Universal content loader that fetches and processes content from any URL.
    Automatically detects content type (webpage, PDF, or image) based on URL.
    """
    return await load_content(url, 20_000, offset, raw)


@mcp.tool()
async def view_website(
    url: Annotated[str, Field(description="The webpage URL to capture.")],
    full_page: Annotated[
        bool,
        Field(
            description="Capture the entire scrollable page instead of just the current viewport.",
        ),
    ] = False,
) -> Image:
    """Capture a rendered screenshot of a webpage using Zendriver.

    Args:
        url: The HTTP(S) URL to navigate to before taking the screenshot.
        full_page: When ``True`` capture the full scroll height; defaults to the
            currently visible viewport when ``False``.

    Returns:
        A PNG ``Image`` payload that can be streamed back to the MCP client.
    """

    return await capture_webpage_screenshot(url, full_page=full_page)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
