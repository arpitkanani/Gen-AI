from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="FAISS is a library for efficient similarity search developed by Facebook AI.", metadata={"source": "ai_notes", "page": 1}),
    Document(page_content="Machine learning models learn patterns from training data automatically.", metadata={"source": "ml_notes", "page": 1}),
    Document(page_content="Neural networks are inspired by the structure of the human brain.", metadata={"source": "dl_notes", "page": 1}),
    Document(page_content="LangChain is a framework for building LLM powered applications.", metadata={"source": "lc_notes", "page": 2}),
    Document(page_content="RAG combines retrieval systems with language model generation.", metadata={"source": "rag_notes", "page": 1}),
    Document(page_content="ChromaDB is an open source vector database for AI applications.", metadata={"source": "db_notes", "page": 1}),
    Document(page_content="Embeddings convert text into numerical vectors for semantic search.", metadata={"source": "emb_notes", "page": 2}),
    Document(page_content="Python is the most popular language for machine learning projects.", metadata={"source": "py_notes", "page": 1}),
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Build FAISS vector store
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever(
    search_type="similarity",   # basic similarity search
    search_kwargs={"k": 3}      # return top 3 results
)

# Standard retriever interface — always .invoke()
results = retriever.invoke("What is used for similarity search?")

print("VectorStore Retriever:")
print(f"Results returned: {len(results)}\n")
for i, doc in enumerate(results):
    print(f"Rank {i+1}: {doc.page_content}")
    print(f"Source: {doc.metadata}\n")