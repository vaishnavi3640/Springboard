import streamlit as st
from dotenv import load_dotenv
import os
import shutil

from utils.loaders import load_documents, clean_metadata
from utils.embeddings import get_embeddings
from utils.vectordb import create_vector_store
from utils.splitters import split_documents

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Document Q&A", layout="wide")

st.title("📚 Document Embedding System")

# Check API Key
groq_key = os.getenv("GROQ_API_KEY")

if groq_key:
    st.success("API Key Loaded Successfully ✅")
else:
    st.error("API Key NOT LOADED ❌")

# Build Vector DB Button
if st.button("Build Vector Database"):

    # Cleanup old DB
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
        st.warning("Old Chroma DB deleted")
    else:
        st.info("No previous DB found")

    with st.spinner("Loading documents..."):
        documents = load_documents()
        st.write(f"Loaded {len(documents)} documents")

    with st.spinner("Cleaning metadata..."):
        documents = clean_metadata(documents)
        st.write("Metadata cleaned")

    with st.spinner("Splitting documents..."):
        chunks = split_documents(documents)
        st.write(f"Created {len(chunks)} chunks")

    with st.spinner("Loading embedding model..."):
        embeddings = get_embeddings()
        st.write("Embedding model loaded")

    with st.spinner("Creating vector database..."):
        vectordb = create_vector_store(chunks, embeddings)
        st.success("Vector database created successfully 🎉")