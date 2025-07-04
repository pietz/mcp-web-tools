import os
import logging
import httpx

from duckduckgo_search import DDGS
import googlesearch

logger = logging.getLogger(__name__)


async def brave_search(query: str, limit: int = 10) -> dict | None:
    """
    Search the web using the Brave Search API.
    Requires BRAVE_SEARCH_API_KEY environment variable.
    """
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        return None

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
            return {
                "results": [
                    {
                        "title": x["title"],
                        "url": x["url"],
                        "description": x["description"],
                    }
                    for x in results
                ],
                "provider": "brave"
            }
    except httpx.HTTPStatusError as e:
        logger.warning(
            f"Brave Search API returned status code {e.response.status_code}"
        )
    except httpx.TimeoutException:
        logger.warning("Brave Search API request timed out")
    except Exception as e:
        logger.warning(f"Error using Brave Search: {str(e)}")

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
            "provider": "google"
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
            "provider": "duckduckgo"
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
    return {"results": [], "provider": "none"}  # Return empty results if all search methods fail
