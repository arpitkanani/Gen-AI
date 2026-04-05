from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

model=ChatOllama(model="llama3.2:3b")

prompt1 = ChatPromptTemplate([
    ("system","write a detail report on  {topic}"),   
    ("human","{topic}")
])

prompt2 = ChatPromptTemplate([
    ("system","write a summary of the report on {topic}"),
    ("human","{topic}")
])
parser=StrOutputParser()
report_chain=prompt1 |model | parser 

# in Branch runnable (condition,chain)
branch_chain=RunnableBranch(
    (lambda x: len(x.split())>100, prompt2 | model | parser), # type: ignore
    RunnableLambda(lambda x: "Report is concise enough, no summary needed.")
)

chain= report_chain  | branch_chain

print(chain.invoke({"topic":"AI in healthcare"}))