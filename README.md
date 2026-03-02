# RAG Pipeline Project

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline using:

- Document Loaders (PDF & Text)
- Text Chunking
- Vector Database (ChromaDB)
- Groq LLM API
- Streamlit Interface
- Source Citation

## Project Structure

Springboard/
│
├── data/              # Input documents
├── utils/             # Loaders and helper modules
├── app.py             # Main Streamlit app
├── requirements.txt   # Dependencies
└── .gitignore

## Milestones Completed

### Milestone 1
- Document loading
- Text splitting
- Embedding generation
- Chroma vector store creation

### Milestone 2
- Retrieval mechanism
- Groq LLM integration
- Answer synthesis
- Source citation display
- Streamlit UI implementation

## How to Run


pip install -r requirements.txt

streamlit run app.py