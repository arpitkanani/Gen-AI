from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
docs = [
    Document(page_content="Neural networks have many layers. Gradient descent is an optimization algorithm that minimizes loss by adjusting weights. Training requires large datasets. Overfitting happens when model memorizes training data.", metadata={"source": "dl_notes", "page": 1}),
    Document(page_content="Python is popular for data science. Machine learning models learn from data. FAISS is used for similarity search. ChromaDB stores embeddings persistently.", metadata={"source": "tools_notes", "page": 1}),
    Document(page_content="RAG stands for Retrieval Augmented Generation. It combines vector search with language models. RAG helps LLMs answer questions about specific documents.", metadata={"source": "rag_notes", "page": 1}),
    Document(page_content="Transformers use attention mechanism. BERT is bidirectional transformer. GPT is autoregressive model. Both are pretrained on large text corpora.", metadata={"source": "nlp_notes", "page": 1}),
    Document(page_content="Embeddings represent text as vectors. Similar texts have similar vectors. Word2Vec and sentence transformers generate embeddings.", metadata={"source": "emb_notes", "page": 1}),
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(docs, embeddings)
base_retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="llama3.2:3b")

# ─────────────────────────────────────────
# LLMChainExtractor — compresses each chunk
# uses LLM to extract only relevant sentences
# ─────────────────────────────────────────
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,     # what compresses the chunks
    base_retriever=base_retriever   # what fetches the chunks
)

query = "What is gradient descent?"
results = compression_retriever.invoke(query)

print("Contextual Compression Retriever:")
print(f"Query: '{query}'\n")
print("--- Normal retriever would return full chunks ---")
normal_results = base_retriever.invoke(query)
for doc in normal_results:
    print(f"FULL: {doc.page_content}\n")

print("--- Compression retriever returns only relevant part ---")
for doc in results:
    print(f"COMPRESSED: {doc.page_content}\n")