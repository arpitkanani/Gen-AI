from langchain_chroma import Chroma
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


chroma_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_index",  # auto saves here ✅
    collection_name="my_docs"          # name your collection
)

query = "What is used for similarity search?"
results_with_scores = chroma_db.similarity_search_with_score(query, k=3)

print(f"\nQuery: {query}")
print("Results with similarity scores:\n")
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")


loaded_chroma = Chroma(
    persist_directory="chroma_index",
    embedding_function=embeddings,
    collection_name="my_docs"
)
print("ChromaDB loaded from disk!")
result = loaded_chroma.similarity_search("what is RAG?", k=1)
print(f"Loaded DB result: {result[0].page_content}")