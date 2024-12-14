# Document Parsing and Retrieval Agent Pipeline

This repository contains a two-agent pipeline for processing, chunking, embedding, and retrieving information from unstructured documents, such as PDFs. The pipeline leverages large language models (LLMs), specifically **Gemini**, for document parsing and question answering, and **ChromaDB** for vector-based retrieval.

## Overview

The pipeline is designed to handle documents with various formats, such as tables, figures, images, and text. The two main steps are:

1. **Document Parsing and Chunking**: Extracts and summarizes key sections (tables, figures, text blocks) from each page of a PDF, leveraging Gemini's capabilities to process and understand mixed content.
   
2. **Document Retrieval and Question Answering**: Vectorizes the summarized chunks and stores them in a vector database (Chroma). When a question is posed, it retrieves the most relevant sections and generates an answer, providing contextual information (including page images) to improve accuracy.

### Features
- **Document parsing**: Extracts images from PDFs and splits the content into relevant sections.
- **Chunking and summarization**: Uses a LLM to categorize content into chunks (e.g., tables, figures, text).
- **Vector-based search**: Utilizes a vector store (ChromaDB) to index and retrieve relevant document chunks.
- **Question answering**: Allows querying of the document, with the response generated using the context of the most relevant chunks and images.

```bash
pip install -r requirements.txt
```

## Setup

### 1. Install and Set up Google Generative AI (Gemini)
The pipeline uses Gemini for both parsing the document and generating responses. You will need access to Gemini's API. Ensure that you have the necessary credentials to use the model.

### 2. Document Parsing Agent

This agent handles the parsing and chunking of the document. It extracts each page of the document as a JPEG image, splits the page into relevant chunks (tables, figures, text), and generates summaries for each chunk. 

The relevant code is located in `document_parsing_agent.py`.


### 3. Document Retrieval and Question Answering Agent

This agent handles indexing the parsed document in a vector store and performing search and retrieval based on user queries. It uses ChromaDB to store the vectorized summaries of each chunk and retrieve the most relevant ones for generating answers.

The relevant code is located in `document_rag_agent.py`.

## Running the Code

1. **Parse and Chunk the Document**: The first agent will extract images, identify chunks, and summarize the content.

2. **Index the Documents**: The second agent will take the parsed document, embed the summaries, and index them in ChromaDB for retrieval.

3. **Ask Questions**: Once the document is indexed, you can query it by asking questions. The system will search for the most relevant chunks and provide an answer using the context from those chunks, along with the images.

### Example Usage

Check notebooks/agents_demo.ipynb

## Future Improvements

1. **Persistent Storage**: Currently, the vector store is in-memory using ChromaDB. In production, consider using persistent storage options like Pinecone or Weaviate.
2. **Real-Time Updates**: Enhance the system to support dynamic document updates (e.g., re-indexing on document modification).
3. **Advanced Retrieval**: Implement hybrid search techniques for better retrieval accuracy by combining dense and sparse embeddings.
4. **Support for More Document Formats**: Expand the system to handle other formats like Word or PowerPoint in addition to PDFs.

## License

This code is licensed under the MIT License. See the LICENSE file for more details.

## Data source

Sample PDF: https://raw.githubusercontent.com/SharifiZarchi/Introduction_to_Machine_Learning/main/Slides/Chapter_05_Natural_Language_Processing/04-LLM%26Adaptation/LLM%20%26%20Adaptation.pdf
Document VQA: https://rrc.cvc.uab.es/?ch=17&com=downloads or https://huggingface.co/datasets/lmms-lab/DocVQA