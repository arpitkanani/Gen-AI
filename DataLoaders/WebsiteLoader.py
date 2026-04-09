import bs4
from langchain_community.document_loaders import WebBaseLoader

# ─────────────────────────────────────────
# BASIC — load any webpage
# ─────────────────────────────────────────
loader = WebBaseLoader(
    web_paths=["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"],
    
    # bs_kwargs tells BeautifulSoup WHICH parts of HTML to extract
    # without this it grabs everything — navbars, footers, ads etc
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer( # type: ignore
            class_=("mw-body-content")  # Wikipedia's main content div
        )
    )
)

docs = loader.load()

print("Total documents:", len(docs))
print("\nContent preview (first 500 chars):")
print(docs[0].page_content[:500])
print("\nMetadata:")
print(docs[0].metadata)

# ─────────────────────────────────────────
# MULTIPLE URLs at once
# ─────────────────────────────────────────
multi_loader = WebBaseLoader(
    web_paths=[
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://en.wikipedia.org/wiki/Word2vec"
    ]
)

multi_docs = multi_loader.load()
print("\n--- Multiple URLs ---")
print("Total documents loaded:", len(multi_docs))
for doc in multi_docs:
    print(f"Source: {doc.metadata['source']} | chars: {len(doc.page_content)}")