from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel,RunnablePassthrough

model=ChatOllama(model="llama3.2:3b")

prompt1=ChatPromptTemplate([
    ("system","Generate a joke about {topic}"),   
    ("human","{topic}")
])
parser = StrOutputParser()


prompt2=ChatPromptTemplate([
    ("system","Generate a explanation of given joke {joke}"),
    ("human","{joke}")
])
joke_chain=prompt1 | model | parser

chain=RunnableParallel({
    "joke":RunnablePassthrough(),
    "explanation":prompt2 | model | parser
})
final_chain=joke_chain | chain

result = final_chain.invoke({"topic":"AI"})
print(result)
