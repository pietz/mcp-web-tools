import asyncio
import logging
import os

import httpx
import googlesearch
from duckduckgo_search import DDGS
from perplexity import PerplexityError, AsyncPerplexity


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

    attempts = 3
    for attempt in range(1, attempts + 1):
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
            status = e.response.status_code if e.response is not None else None
            if status == 429 and attempt < attempts:
                logger.info("Brave rate limited (429); retrying in 1s (%d/%d)", attempt, attempts)
                await asyncio.sleep(1)
                continue
            logger.warning(
                f"Brave Search API returned status code {status}"
            )
            break
        except httpx.TimeoutException:
            logger.warning("Brave Search API request timed out")
            break
        except Exception as e:
            logger.warning(f"Error using Brave Search: {str(e)}")
            break

    return None


async def perplexity_search(query: str, limit: int = 10) -> dict | None:
    """
    Search the web using the Perplexity Search API.
    Returns None if no API key is configured or on failure.
    """

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return None

    client = AsyncPerplexity(api_key=api_key)

    try:
        response = await client.search.create(query=query, max_results=limit)
    except PerplexityError as exc:
        logger.warning("Perplexity Search API failed: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected Perplexity Search error: %s", exc)
        return None

    if not getattr(response, "results", None):
        return None

    mapped_results = [
        {
            "title": result.title,
            "url": result.url,
            "description": result.snippet,
        }
        for result in response.results
        if getattr(result, "title", None) and getattr(result, "url", None)
    ]

    if not mapped_results:
        return None

    return {"provider": "perplexity", "results": mapped_results}


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
    Tries Perplexity Search first (if API key available), then Brave, Google, finally DuckDuckGo.
    Returns a dictionary with search results and the provider used.
    """
    # Try Perplexity Search first
    results = await perplexity_search(query, limit)
    if results:
        return results

    # Try Brave Search next
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
    return {"results": [], "provider": "none"}
