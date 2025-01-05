from document_ai_agents.tools import WikipediaSearchResponse, search_wikipedia


def test_wikipedia_search_tool():
    result = search_wikipedia(search_query="Stevia")

    assert isinstance(result, WikipediaSearchResponse)
