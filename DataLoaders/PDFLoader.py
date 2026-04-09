from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

model=ChatOllama(model="llama3.2:3b")
prompt1=ChatPromptTemplate([
    ("system","Act as a Pro Report summary and KPI finders"),
    ("read the data and generate what data asked {data}")
])

parser=StrOutputParser()
chain=prompt1 | model | parser

loader = PyPDFLoader("C:\\Users\\kanan\\OneDrive\\Desktop\\RAG\\LangChain\\DataLoaders\\test.pdf")
docs = loader.load()

print("Total pages loaded:", len(docs))


print(chain.invoke({"data":docs[0]}))