from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

model=ChatOllama(model="llama3.2:3b")

prompt1=ChatPromptTemplate([
    ("system","Generate a tweet about {topic}"),
    ("human","{topic}")
])

parser = StrOutputParser()
prompt2=ChatPromptTemplate([
    ("system","Generate a LinkedIn post about {topic}"),
    ("human","{topic}")
])

parllel_chain=RunnableParallel({
    "tweet":prompt1 | model | parser,
    'linkedin': prompt2 | model | parser
})
result=parllel_chain.invoke({"topic":"AI in healthcare"})
print("Tweet:", result["tweet"])
print("LinkedIn Post:", result["linkedin"])