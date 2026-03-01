import streamlit as st
from dotenv import load_dotenv
import os
import shutil

from langchain_groq import ChatGroq

from utils.loaders import load_documents, clean_metadata
from utils.embeddings import get_embeddings
from utils.vectordb import create_vector_store
from utils.splitters import split_documents


# LOAD ENV VARIABLES
load_dotenv()

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("📚 RAG Assistant")


# CHECK GROQ API KEY
groq_key = os.getenv("GROQ_API_KEY")

if groq_key:
    st.success("Groq API Key Loaded ✅")
else:
    st.error("Groq API Key NOT Loaded ❌")


# SESSION STATE INIT
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None


# BUILD VECTOR DATABASE
if st.button("Build Database"):

    documents = load_documents()
    documents = clean_metadata(documents)

    st.write("Loaded documents:", len(documents))

    # ✅ Show unique sources only
    sources = set(doc.metadata.get("source") for doc in documents)
    for src in sources:
        st.write("Loaded:", src)

    chunks = split_documents(documents)
    embeddings = get_embeddings()

    vectordb = create_vector_store(chunks, embeddings)

    st.session_state.vectordb = vectordb

    st.success("Vector Database Created Successfully 🎉")
# USER QUERY SECTION
user_query = st.text_input("Ask a question from your documents:")

if user_query:

    if st.session_state.vectordb is None:
        st.warning("Please build the vector database first.")

    else:
        retriever = st.session_state.vectordb.as_retriever(
            search_kwargs={"k": 2}
        )

        docs = retriever.invoke(user_query)

        # Combine retrieved context
        context = "\n\n".join([doc.page_content for doc in docs])

        # LLM INITIALIZATION
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        prompt = f"""
You are a helpful assistant.

Use ONLY the information provided in the context below to answer the question.
If the answer is not present in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{user_query}

Answer in a clear and concise paragraph.
"""

        response = llm.invoke(prompt)

        st.write("## 📌 Final Answer:")
        st.write(response.content)

        # SOURCE CITATION (FIXED POSITION)
        st.write("### 📖 Sources:")

        if docs:
            sources = list(set(
                doc.metadata.get("source", "Unknown Source")
                for doc in docs
            ))

            for source in sources:
                st.write(f"- {source}")
        else:
            st.write("No sources found.")