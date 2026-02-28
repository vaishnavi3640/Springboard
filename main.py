from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredPDFLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_.env()
if os.path.exists("chroma_db"):python main.py
    shutil.rmtree("chroma_db")

print("Loading documents...")

pdf_loader = DirectoryLoader(
    "data",
    glob="**/*.pdf",
    loader_cls=UnstructuredPDFLoader
)

txt_loader = DirectoryLoader(
    "data",
    glob="**/*.txt",
    loader_cls=TextLoader
)

documents = pdf_loader.load() + txt_loader.load()

print(f"Loaded {len(documents)} documents.")

print("Splitting documents into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

cleaned_chunks = []

for chunk in chunks:
    chunk.metadata = {k: str(v) for k, v in chunk.metadata.items()}
    cleaned_chunks.append(chunk)

print("Initializing embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating vector database...")

collection_name = f"collection_{uuid.uuid4()}"

vectorstore = Chroma.from_documents(
    documents=cleaned_chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db",
    collection_name=collection_name
)

vectorstore.persist()

print("Vector database created successfully.")

print("Initializing Groq model...")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff"
)
print("\nSystem Ready ✅")
print("You can now ask questions from your documents.\n")

while True:
    query = input("Ask a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    response = qa_chain.invoke(query)

    print("\nAnswer:\n")
    print(response["result"])
    print("\n" + "="*50 + "\n")