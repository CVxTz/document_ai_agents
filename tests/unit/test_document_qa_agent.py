from pathlib import Path

from document_ai_agents.document_qa_agent import DocumentQAAgent, DocumentQAState
from document_ai_agents.document_utils import extract_images_from_pdf
from document_ai_agents.image_utils import pil_image_to_base64_jpeg


def test_document_qa_agent():
    document_path = str(Path(__file__).parents[2] / "data" / "docs.pdf")

    images = extract_images_from_pdf(pdf_path=document_path)
    pages_as_base64_jpeg_images = [pil_image_to_base64_jpeg(x) for x in images]

    state2 = DocumentQAState(
        question="What is the highest score on M-RCNN ?",
        pages_as_base64_jpeg_images=pages_as_base64_jpeg_images,
    )

    agent = DocumentQAAgent()

    result = agent.graph.invoke(state2)

    assert result["answer_cot"].answer == "0.708"
    assert result["verification_cot"].verified


def test_document_qa_agent_text():
    state2 = DocumentQAState(
        question="Who is the 20th president of the US?",
        pages_as_base64_jpeg_images=[],
        pages_as_text=[
            "James Garfield was elected as the United States' 20th President in 1880, after nine terms in "
            "the U.S. House of Representatives."
        ],
    )

    agent = DocumentQAAgent()

    result = agent.graph.invoke(state2)

    assert result["answer_cot"].answer == "James Garfield"
    assert result["verification_cot"].verified
