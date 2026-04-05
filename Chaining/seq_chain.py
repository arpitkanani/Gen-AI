from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
model=ChatOllama(model="llama3.2:1b")
model1=OllamaLLM(model="llama3.2:1b")
prompt1=PromptTemplate(
    template="generate a detailed report on {topic}",
    input_variables=["topic"]
)
prompt2=PromptTemplate(
    template="Generate 5 point each one line summary on given text \n {text}",
    input_variables=["text"]
)
parser=StrOutputParser()

chain=prompt1 | model1 | parser | prompt2 | model |parser

result=chain.invoke({"topic":"ISRO"})


print(result)