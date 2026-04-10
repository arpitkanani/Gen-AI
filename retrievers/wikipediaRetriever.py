from langchain_community.retrievers import WikipediaRetriever


# searches Wikipedia directly, no vector DB

retriever = WikipediaRetriever(
    top_k_results=2,        # how many Wikipedia articles
    doc_content_chars_max=500  # max chars per article
) # type: ignore

results = retriever.invoke("What is Retrieval Augmented Generation?")

print("Wikipedia Retriever:")
print(f"Articles found: {len(results)}\n")
for i, doc in enumerate(results):
    print(f"Article {i+1}:")
    print(f"Content: {doc.page_content[:300]}")
    print(f"Metadata: {doc.metadata}\n")