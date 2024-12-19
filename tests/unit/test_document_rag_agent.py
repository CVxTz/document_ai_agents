from pathlib import Path

from document_ai_agents.document_parsing_agent import (
    DocumentLayoutParsingState,
    DocumentParsingAgent,
)
from document_ai_agents.document_rag_agent import DocumentRAGAgent, DocumentRAGState


def test_rag_agent():
    state1 = DocumentLayoutParsingState(
        document_path=str(Path(__file__).parents[2] / "data" / "docs.pdf")
    )

    agent1 = DocumentParsingAgent()

    result1 = agent1.graph.invoke(state1)

    state2 = DocumentRAGState(
        question="Who was acknowledge in this paper ?",
        document_path=str(Path(__file__).parents[1] / "data" / "docs.pdf"),
        pages_as_base64_jpeg_images=result1["pages_as_base64_jpeg_images"],
        documents=result1["documents"],
    )

    agent2 = DocumentRAGAgent()

    result2 = agent2.graph.invoke(state2)

    assert "Manoj" in result2["response"]
