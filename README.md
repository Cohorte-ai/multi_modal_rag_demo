# Multimodal RAG

## Overview

This project used Multimodal RAG (Retrieval-Augmented Generation) technique to extract, process, and query information from PDF documents through a seamless integration of Optical Character Recognition (OCR), Natural Language Processing (NLP), and innovative machine learning technologies. This project harnesses the power of state-of-the-art AI models to provide a robust platform for information retrieval and content understanding across various formats.

## Demo
You checkout the demo at https://drive.google.com/file/d/17DkFPHm3OQdsiEBXUUesDRBpxFJ0-rz8/view?usp=sharing

## Features

- **PDF and Image Processing**: Automated extraction of text and imagery from PDF files, converting all accessible information into machine-readable formats.
- **Image-to-Text Conversion**: Utilizes the OpenAI API for transforming images into descriptive text, enabling further analysis and indexing.
- **Advanced Text and Image Indexing**: Leverages the Chroma vector database for embedding and efficient similarity searching of text and image-derived content.
- **Interactive Query Interface**: A user-friendly web interface built with Streamlit, allowing for intuitive searching and exploration of indexed content.

## Technical Components

- **LangChain**: Orchestrates document loading, text extraction, splitting, and the integration of NLP workflows.
- **OpenAI APIs (GPT and Vision Models)**: For high-accuracy image-to-text conversion and generating text embeddings that capture deep semantic meanings.
- **Chroma**: Employs a vector-based approach to store and retrieve text embeddings, facilitating fast and relevant similarity searches.
- **Streamlit**: Powers the interactive web interface, making it easy for users to query the database and receive instant, relevant responses.

## Machine Learning Concepts

### Optical Character Recognition (OCR)
Utilized for converting different types of images containing textual information into machine-encoded text, enabling digital processing and analysis.

### Natural Language Processing (NLP)
Involves the use of algorithms to understand and manipulate human language. This project employs NLP to analyze, understand, and generate human language from the text extracted from documents and images.

### Embeddings and Vector Databases
Transforms text into high-dimensional vectors or embeddings, allowing the comparison of semantic similarities between pieces of text. Chroma, a vector database, is used to efficiently store and query these embeddings.

### Retrieval-Augmented Generation (RAG)
Combines the retrieval of relevant documents or text chunks with state-of-the-art language generation models. This approach enhances the quality and relevance of generated text responses to queries, by grounding them in the content extracted from the multimodal data sources (PDFs and images).

### Multimodal Learning
Integrates and processes information from various types of data, such as text and images, enhancing the model's understanding and capability to generate comprehensive and contextually relevant responses.

## Setup

1. Clone the repository.
2. Install required dependencies: `pip install -r requirements.txt`.
3. Configure the `.env` file with necessary environment variables (`OPENAI_API_KEY`).

## Usage

- Execute `python create_database.py` to process the documents, images, and populate the database.
- Launch the Streamlit application: `streamlit run app.py`.
- Visit the provided URL to start querying the system.
