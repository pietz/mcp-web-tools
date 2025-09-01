import os
import logging

from duckduckgo_search import DDGS
import googlesearch
from brave_search_python_client import BraveSearch, WebSearchRequest

logger = logging.getLogger(__name__)


async def brave_search(query: str, limit: int = 10) -> dict | None:
    """
    Search the web using the Brave Search API.
    Requires BRAVE_SEARCH_API_KEY environment variable.
    """
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        return None

    logger.info("Using Brave Search (SDK)...")
    try:
        client = BraveSearch()
        req = WebSearchRequest(q=query, count=limit)
        res = await client.web(req, retries=3, wait_time=1)
        if not res.web or not res.web.results:
            return None
        return {
            "results": [
                {"title": r.title, "url": r.url, "description": r.description}
                for r in res.web.results
            ],
            "provider": "brave",
        }
    except Exception as e:
        logger.warning(f"Error using Brave SDK: {e}")
    return None


def google_search(query: str, limit: int = 10) -> dict | None:
    """
    Search the web using Google Search.
    """
    try:
        logger.info("Using Google Search...")
        results = googlesearch.search(query, num_results=limit, advanced=True)
        if not results:
            raise ValueError("No results returned from Google Search")
        return {
            "results": [
                {"title": r.title, "url": r.url, "description": r.description}
                for r in results
            ],
            "provider": "google",
        }
    except Exception as e:
        logger.warning(f"Error using Google Search: {str(e)}")
    return None


def duckduckgo_search(query: str, limit: int = 10) -> dict | None:
    """
    Search the web using DuckDuckGo.
    """
    try:
        logger.info("Using DuckDuckGo Search...")
        results = list(DDGS().text(query, max_results=limit))
        if not results:
            raise ValueError("No results returned from DuckDuckGo")
        return {
            "results": [
                {"title": r["title"], "url": r["href"], "description": r["body"]}
                for r in results
            ],
            "provider": "duckduckgo",
        }
    except Exception as e:
        logger.warning(f"Error using DuckDuckGo: {str(e)}")
    return None


async def web_search(query: str, limit: int = 10, offset: int = 0) -> dict:
    """
    Search the web using multiple providers, falling back if needed.
    Tries Brave Search API first (if API key available), then Google, finally DuckDuckGo.
    Returns a dictionary with search results and the provider used.
    """
    # Try Brave Search first
    results = await brave_search(query, limit)
    if results:
        return results

    # Fall back to Google
    results = google_search(query, limit)
    if results:
        return results

    # Fall back to DuckDuckGo
    results = duckduckgo_search(query, limit)
    if results:
        return results

    logger.error("All search methods failed.")
    return {
        "results": [],
        "provider": "none",
    }
