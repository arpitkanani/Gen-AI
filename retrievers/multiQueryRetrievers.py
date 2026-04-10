from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


docs = [
    Document(page_content="Supervised learning uses labeled data to train models.", metadata={"source": "ml_notes", "page": 1}),
    Document(page_content="Unsupervised learning finds hidden patterns in unlabeled data.", metadata={"source": "ml_notes", "page": 2}),
    Document(page_content="Reinforcement learning trains agents through rewards and penalties.", metadata={"source": "ml_notes", "page": 3}),
    Document(page_content="Neural networks consist of layers of interconnected nodes.", metadata={"source": "dl_notes", "page": 1}),
    Document(page_content="Gradient descent optimizes model weights during training.", metadata={"source": "dl_notes", "page": 2}),
    Document(page_content="Overfitting occurs when model memorizes training data too well.", metadata={"source": "ml_notes", "page": 4}),
    Document(page_content="Transfer learning reuses pretrained models for new tasks.", metadata={"source": "dl_notes", "page": 3}),
    Document(page_content="BERT and GPT are popular transformer based language models.", metadata={"source": "nlp_notes", "page": 1}),
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(docs, embeddings)
base_retriever = db.as_retriever(search_kwargs={"k": 2})

llm = ChatOllama(model="llama3.2:3b")

# MULTI QUERY RETRIEVER

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# ambiguous query — MultiQuery handles it well
query = "tell me about AI learning"
results = multiquery_retriever.invoke(query)

print("MultiQuery Retriever:")
print(f"Original query: '{query}'")
print(f"Unique results found: {len(results)}\n")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content}")
    print(f"   Source: {doc.metadata}\n")