from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="Machine learning is a subset of AI that learns from data.", metadata={"source": "ml_notes", "page": 1}),
    Document(page_content="ML algorithms learn patterns automatically from training data.", metadata={"source": "ml_notes", "page": 2}),
    Document(page_content="Machine learning models improve their performance over time.", metadata={"source": "ml_notes", "page": 3}),
    Document(page_content="Neural networks are a popular type of machine learning model.", metadata={"source": "dl_notes", "page": 1}),
    Document(page_content="Deep learning uses multiple layers of neural networks.", metadata={"source": "dl_notes", "page": 2}),
    Document(page_content="ML is used in image recognition, NLP, and recommendation systems.", metadata={"source": "app_notes", "page": 1}),
    Document(page_content="Python and PyTorch are popular tools for machine learning.", metadata={"source": "tools_notes", "page": 1}),
    Document(page_content="Supervised learning uses labeled data to train ML models.", metadata={"source": "ml_notes", "page": 4}),
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(docs, embeddings)

normal_retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
normal_results = normal_retriever.invoke("What is machine learning?")

print("Normal Similarity Search:")
for i, doc in enumerate(normal_results):
    print(f"  {i+1}. {doc.page_content}")

# lambda controls relevance vs diversity balance
# lambda=1.0 → pure similarity (same as normal)
# lambda=0.0 → pure diversity
# lambda=0.5 → balanced (recommended)
mmr_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,      # fetch 10 candidates first
        "lambda_mult": 0.3  # balance relevance + diversity
    }
)
mmr_results = mmr_retriever.invoke("What is machine learning?")

print("\nMMR Search (diverse results):")
for i, doc in enumerate(mmr_results):
    print(f"  {i+1}. {doc.page_content}")

print("\nCompare — MMR should be more diverse than normal!")