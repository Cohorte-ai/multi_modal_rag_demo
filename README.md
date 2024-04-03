# Multimodal RAG

This project leverages the power of multimodal data processing to create and query a knowledge base. The goal is to extract meaningful information from PDF documents and provide relevant answers to user queries based on the document content. This README outlines the core concepts, components, and steps involved in setting up and using the project.

## Overview

The project consists of two main scripts: `create_database.py` and `query_data.py`, along with a `requirements.txt` file for dependency management. The system integrates text extraction, image processing, and semantic embeddings to generate a searchable knowledge database from a collection of PDF documents.

### Core Concepts

- **Multimodal Data Processing**: The project handles both text and image data extracted from PDF files, acknowledging the diverse nature of document content.
- **Document Embedding**: It transforms extracted text into high-dimensional vectors using OpenAI embeddings. This representation facilitates semantic search capabilities.
- **Knowledge Database**: Utilizes Chroma, a vector storage solution, to store and efficiently query the embedded documents.
- **Semantic Search**: Answers queries by finding the most relevant document segments based on semantic similarity to the query text.

### Components

1. **Document Loaders**:
   - **DirectoryLoader**: Scans a specified directory for PDF files to process.
   - **PyMuPDFLoader & PyMuPDFLoaderImage**: Extracts text and optionally images from PDF documents.

2. **Text Splitter**:
   - **RecursiveCharacterTextSplitter**: Splits document text into manageable chunks for processing and embedding.

3. **Embeddings**:
   - **OpenAIEmbeddings**: Generates semantic embeddings for text chunks, facilitating similarity searches.

4. **Vector Storage (Chroma)**:
   - Stores document embeddings in a searchable format, allowing for efficient similarity-based queries.

5. **Query and Response Generation**:
   - Uses a similarity search to find the most relevant document chunks to a user query and generates a contextual prompt for a language model to answer the query.

### Setup

1. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare your environment:
    Place your PDF documents in a directory named data/raw.
    Ensure you have a .env file with necessary configurations (e.g., API keys).
3. Generating the Knowledge Database
 Run create_database.py to load documents from the specified directory, extract text (and optionally images), embed the text, and save the data into Chroma for future querying.
4. Querying the Database
Run query_data.py with your query text to search the knowledge database and get relevant information based on the content of the loaded documents.
5. Dependencies
See requirements.txt for a list of Python packages required to run the project.

