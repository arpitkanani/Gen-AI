from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

## Recusrive text splitter and take less time as character text splitter 
loader = PyPDFLoader("test.pdf")
docs = loader.load()

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

recursive_chunks = recursive_splitter.split_documents(docs)

print("\nRecursiveCharacterTextSplitter:")
print(f"Total chunks: {len(recursive_chunks)}")
print(f"\nChunk 1:\n{recursive_chunks[0].page_content}")
print(f"\nMetadata preserved: {recursive_chunks[0].metadata}")


print("\nOverlap visualization:")
print(f"End of chunk 1:\n...{recursive_chunks[0].page_content[-80:]}")
print(f"\nStart of chunk 2:\n{recursive_chunks[1].page_content[:80]}...")


print("\nChunk size comparison:")
print(f"Original page 1 size : {len(docs[0].page_content)} chars")
print(f"After splitting      : {len(recursive_chunks)} total chunks")
print(f"Average chunk size   : {sum(len(c.page_content) for c in recursive_chunks) // len(recursive_chunks)} chars")

