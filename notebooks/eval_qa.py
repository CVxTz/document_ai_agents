from pathlib import Path

from document_ai_agents.document_qa_agent import DocumentQAAgent, DocumentQAState
from document_ai_agents.image_utils import image_file_to_base64_jpeg

if __name__ == "__main__":
    image_path = str(
        Path(__file__).parents[1]
        / "data"
        / "docvqa"
        / "spdocvqa_images"
        / "hxxl0226_1.png"
    )

    pages_as_base64_jpeg_images = [image_file_to_base64_jpeg(image_path)]

    state = DocumentQAState(
        # question="What is the milk substitute discussed in this document?",
        question="What day of the week was this invoice received?",
        pages_as_base64_jpeg_images=pages_as_base64_jpeg_images,
        pages_as_text=[],
    )

    agent = DocumentQAAgent()

    result = agent.graph.invoke(state)

    print(result["answer_cot"])
    print(result["verification_cot"])
