from langchain_community.document_loaders import PyPDFLoader
loader= PyPDFLoader('vi.pdf')
docs=loader.load()
print(len(docs))