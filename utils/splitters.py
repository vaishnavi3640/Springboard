from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitters = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )

    return splitters.split_documents(documents)