import pytest
from mcp_web_tools.search import web_search


@pytest.mark.asyncio
async def test_provider_flag_in_response():
    """Integration test to verify that the provider flag is correctly added to search results."""
    result = await web_search("test query", limit=3)
    
    # Check that result has the expected structure
    assert isinstance(result, dict)
    assert "results" in result
    assert "provider" in result
    
    # Check that provider is one of the expected values
    assert result["provider"] in ["brave", "google", "duckduckgo", "none"]
    
    # Check that results is a list
    assert isinstance(result["results"], list)
    
    # If we have results, check their structure
    if result["results"]:
        for item in result["results"]:
            assert "title" in item
            assert "url" in item
            assert "description" in item
    
    print(f"\nSearch provider used: {result['provider']}")
    print(f"Number of results: {len(result['results'])}")