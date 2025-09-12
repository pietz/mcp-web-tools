import os
import logging

from duckduckgo_search import DDGS
import googlesearch
import httpx


logger = logging.getLogger(__name__)


async def brave_search(query: str, limit: int = 10) -> dict | None:
    """
    Search the web using the Brave Search API via a naive HTTPX call.
    Returns None if no API key is configured or on any failure.
    """
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return None

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"X-Subscription-Token": api_key}
    params = {"q": query, "count": limit}

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=headers, params=params, timeout=10)
            r.raise_for_status()
            results = r.json()["web"]["results"]
            return {
                "provider": "brave",
                "results": [
                    {
                        "title": x["title"],
                        "url": x["url"],
                        "description": x.get("description", ""),
                    }
                    for x in results
                ],
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
        # Log and allow fallback to other providers
        logger.warning(f"Google search failed: {str(e)}")
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
        # Log and allow fallback to other providers
        logger.warning(f"DuckDuckGo search failed: {str(e)}")
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
    
    logging.error("All search providers failed.")
    return {"error": "All search providers failed."}
