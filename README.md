# Multimodal RAG

## Overview

This project used Multimodal RAG (Retrieval-Augmented Generation) technique to extract, process, and query information from PDF documents through a seamless integration of Optical Character Recognition (OCR), Natural Language Processing (NLP), and state-of-the-art machine learning technologies to provide a robust platform for information retrieval and content understanding. It offers flexibility in image-to-text conversion by supporting both OpenAI's API and the LLaVA model from Hugging Face, catering to diverse requirements and use cases.

## Features

- **PDF and Image Processing**: Automated extraction of text and imagery from PDF files, making all accessible information machine-readable.
- **Adaptive Image-to-Text Conversion**: Users can choose between the OpenAI API or the LLaVA model for converting images to descriptive text, enabling deep analysis and indexing.
- **Advanced Text and Image Indexing**: Leverages the Chroma vector database for embedding and efficiently searching through text and image-derived content.
- **Interactive Query Interface**: A user-friendly web interface built with Streamlit, facilitating intuitive searches and content exploration.

## Technical Components

- **LangChain**: Manages document loading, text extraction, splitting, and the integration of complex NLP workflows.
- **OpenAI API / LLaVA Model**: Provides options for high-accuracy image-to-text conversion through OpenAI's models or the "llava-hf/llava-1.5-7b-hf" model, capable of understanding and generating descriptions for complex visual content.
- **Chroma**: Utilizes a vector-based approach for storing and retrieving text embeddings, ensuring quick and relevant similarity searches.
- **Streamlit**: Enhances user interaction through a web interface, enabling straightforward querying and result visualization.

## Machine Learning Concepts

### Optical Character Recognition (OCR)
Facilitates the conversion of textual content from images into machine-encoded text for further digital processing.

### Natural Language Processing (NLP)
Employs algorithms to analyze, understand, and generate human language from the text extracted from documents and images.

### Embeddings and Vector Databases
Converts text into high-dimensional vectors or embeddings, enabling semantic similarity comparisons.

### Retrieval-Augmented Generation (RAG)
Improves the quality and relevance of text responses by basing them on content extracted from multimodal data sources.

### Multimodal Learning
Processes and analyzes information from varied data types, such as text and images, to improve overall understanding and contextually relevant responses.

### Flexible Image-to-Text Conversion
Offers the use of either OpenAI's cutting-edge models or the LLaVA model for image description tasks, supporting various project requirements and enhancing the system's adaptability.

## Setup

1. Clone the repository.
2. Install the required dependencies: `pip install -r requirements.txt`. Or if you want to use LLaVA, install from requirements.llava.txt as well. 
3. Configure the `.env` file with necessary environment variables (`OPENAI_API_KEY`).

## Usage

- Execute `python create_database.py` to process documents and images, then populate the database.
- Start the Streamlit application with `streamlit run app.py`.
- Access the web interface via the provided URL to begin querying the system.
