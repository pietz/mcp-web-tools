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
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=20,
            description="Number of results (max 20).",
        ),
    ] = 5,
    offset: Annotated[
        int,
        Field(
            ge=0,
            description="To scroll through more results.",
        ),
    ] = 0,
) -> dict:
    """
    Execute a web search using the given search query.
    Returns a list of results including title, URL, and a rich content snippet.
    
    """
    return await web_search(query, limit, offset)


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
) -> Image:
    """
    Capture and return a rendered screenshot of a website.
    Helpful when not just the data but also the visuals are relevant.
    """

    return await capture_webpage_screenshot(url, full_page=True)


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
