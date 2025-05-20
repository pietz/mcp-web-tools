import logging
from pydantic import Field

from mcp.server.fastmcp import FastMCP, Image

from .search import search_web
from .loaders import load_webpage, load_pdf_document, load_image_file

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    return await search_web(query, limit, offset)


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
    return await load_webpage(url, limit, offset, raw)


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
    return await load_pdf_document(url, limit, offset, raw)


@mcp.tool()
async def load_image(
    url: str = Field(description="The remote image file URL to fetch."),
) -> Image:
    """
    Fetch an image from the internet and view it.
    """
    return await load_image_file(url)


def main():
    """Entry point for the package when installed via pip."""
    mcp.run()


if __name__ == "__main__":
    main()