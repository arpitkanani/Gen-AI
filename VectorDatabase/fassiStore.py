from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# ─────────────────────────────────────────
# SAMPLE DOCUMENTS (no PDF needed)
# ─────────────────────────────────────────
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

# EMBEDDINGS — converts text to vectors
# using local ollama embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# BUILD FAISS — embeds + stores all docs

faiss_db = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)

query = "What is used for similarity search?"
results = faiss_db.similarity_search(query, k=3)

print(f"\nQuery: {query}")
print(f"Top {len(results)} results:\n")
for i, doc in enumerate(results):
    print(f"Rank {i+1}: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")


faiss_db.save_local("faiss_index")
print("FAISS index saved to disk!")

loaded_db = FAISS.load_local(
    "faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True  # required flag for loading
)
print("FAISS index loaded from disk!")

result = loaded_db.similarity_search("what is langchain?", k=1)
print(f"\nLoaded DB search result: {result[0].page_content}")