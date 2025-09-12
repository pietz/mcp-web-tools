import os

from duckduckgo_search import DDGS
import googlesearch
# Import submodules directly to avoid __init__.py side effects (boot() config)
from brave_search_python_client.client import BraveSearch
from brave_search_python_client.requests import WebSearchRequest


async def brave_search(query: str, limit: int = 10) -> dict | None:
    """
    Search the web using the Brave Search API.
    Requires BRAVE_SEARCH_API_KEY environment variable.
    """
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        return None

    try:
        client = BraveSearch()
        req = WebSearchRequest(q=query, count=limit)
        res = await client.web(req, retries=3, wait_time=1)
        if not res.web or not res.web.results:
            return None
        return {
            "provider": "brave",
            "results": [
                {"title": r.title, "url": r.url, "description": r.description}
                for r in res.web.results
            ],
        }
    except Exception as e:
        # Swallow errors to allow fallback to other providers
        _ = e
    return None


def google_search(query: str, limit: int = 10) -> dict | None:
    """
    Search the web using Google Search.
    """
    try:
        results = googlesearch.search(query, num_results=limit, advanced=True)
        
        # Convert results to list to properly handle generators/iterators
        results_list = list(results) if results else []
        
        # Return None if no results to trigger fallback
        if not results_list:
            return None
            
        return {
            "provider": "google",
            "results": [
                {"title": r.title, "url": r.url, "description": r.description}
                for r in results_list
            ],
        }
    except Exception as e:
        # Swallow errors to allow fallback to other providers
        _ = e
    return None


def duckduckgo_search(query: str, limit: int = 10) -> dict | None:
    """
    Search the web using DuckDuckGo.
    """
    try:
        results = list(DDGS().text(query, max_results=limit))
        if not results:
            raise ValueError("No results returned from DuckDuckGo")
        return {
            "provider": "duckduckgo",
            "results": [
                {"title": r["title"], "url": r["href"], "description": r["body"]}
                for r in results
            ],
        }
    except Exception as e:
        # Swallow errors to allow fallback to other providers
        _ = e
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

    return {
        "provider": "none",
        "results": [],
    }
