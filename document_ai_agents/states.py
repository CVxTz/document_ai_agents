from typing import Literal

from pydantic import BaseModel


class DetectedLayoutElement(BaseModel):
    caption: str
    ymin: int
    xmin: int
    ymax: int
    xmax: int
    element_type: Literal["Table", "Figure", "Image"]


class DetectedLayoutElements(BaseModel):
    layout_elements: list[DetectedLayoutElement] = []


class LayoutElement(BaseModel):
    caption: str
    ymin: float
    xmin: float
    ymax: float
    xmax: float
    page_number: int
    element_type: Literal["Table", "Figure", "Image"]


class DocumentLayoutParsingState(BaseModel):
    document_path: str
    pages_as_images: list = []
    pages_as_texts: list = []
    layout_elements: list[LayoutElement] = []


if __name__ == "__main__":
    state = DocumentLayoutParsingState(document_path="A.pdf")

    print(state.document_path)