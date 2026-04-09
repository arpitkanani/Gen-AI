from langchain_community.document_loaders import TextLoader

loader=TextLoader("test.txt",encoding='utf-8')

docs=loader.load()

print(type(docs))                        # <class 'list'>
print(len(docs))                         # 1 → whole file = 1 document
print(type(docs[0]))                     # <class 'Document'>
print(docs[0].page_content)             # actual text
print(docs[0].metadata)                 # metadata (empty in this case)
