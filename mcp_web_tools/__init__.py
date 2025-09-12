from pydantic import Field

from mcp.server.fastmcp import FastMCP

from .search import web_search
from .loaders import load_content

# Create the MCP server
mcp = FastMCP("Web Tools", log_level="INFO")


@mcp.prompt()
def help() -> str:
    """Load detailed information about the server and its usage."""
    return """
    ## Summary
    This server provides tools for web searching and content extraction.

    ## Usage
    1. Use `web_search` to find potentially relevant URLs based on your query.
    2. Use `fetch_url` to fetch and extract content from any URL - it auto-detects content type.

    ## Notes
    - Rely on unbiased and trusted sources to retrieve accurate results.
    - Use `raw` only if the Markdown extraction fails or to inspect the raw HTML.
    - Images are automatically resized to fit within 1024x1024 dimensions.
    """


@mcp.tool()
async def search_web(
    query: str = Field(description="The search query to use."),
    offset: int = Field(0, ge=0, description="To scroll through more results."),
) -> dict:
    """
    Execute a web search using the given search query and returns 10 results.
    Tries to use Brave first, then Google, finally DuckDuckGo as fallbacks.
    Returns a dictionary with 'results' (list of search results) and 'provider' (search engine used).
    """
    return await web_search(query, 10, offset)


@mcp.tool()
async def fetch_url(
    url: str = Field(description="The remote URL to load content from."),
    offset: int = Field(
        0,
        ge=0,
        description="Character/content offset to start from (for text content).",
    ),
    raw: bool = Field(
        False,
        description="Return raw content instead cleaning it. Relevant for webpages and PDFs.",
    ),
):
    """
    Universal content loader that fetches and processes content from any URL.
    Automatically detects content type (webpage, PDF, or image) based on URL.
    """
    return await load_content(url, 20_000, offset, raw)


def main():
    """Entry point for the package when installed via pip."""
    mcp.run()


if __name__ == "__main__":
    main()
