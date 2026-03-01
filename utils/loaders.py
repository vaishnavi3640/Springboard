from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
import os
print("Files inside data folder:", os.listdir("data"))

def load_documents():
    pdf_loader = DirectoryLoader(
        "data",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )

    text_loader = DirectoryLoader(
        "data",
        glob="**/*.txt",
        loader_cls=TextLoader ,
        loader_kwargs={"encoding":"utf-8"}
    )

    pdf_docs = pdf_loader.load()
    text_docs = text_loader.load()

    return pdf_docs + text_docs


def clean_metadata(documents):
    for doc in documents:
        doc.metadata = {
            key: str(value)
            for key, value in doc.metadata.items()
        }
    return documents