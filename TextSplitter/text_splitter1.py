from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)

loader = PyPDFLoader("test.pdf")
docs = loader.load()
print(f"Pages loaded: {len(docs)}")
print(f"Characters in page 1: {len(docs[0].page_content)}")

char_splitter = CharacterTextSplitter(
    chunk_size=500,       # max 500 characters per chunk
    chunk_overlap=50,     # 50 chars repeated between chunks
    separator="\n"        # split at newlines
)

char_chunks = char_splitter.split_documents(docs)

print("\nCharacterTextSplitter:")
print(f"Total chunks: {len(char_chunks)}")
print(f"\nChunk 1 content:\n{char_chunks[0].page_content}")
print(f"\nChunk 1 metadata: {char_chunks[0].metadata}")
print(f"\nChunk 2 content:\n{char_chunks[1].page_content}")
