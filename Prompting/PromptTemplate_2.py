from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage


llm = ChatOllama(model="llama3.2:1b")


# from_messages takes a list of (role, content) tuples
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} teacher. Always give examples."),
    ("human", "Explain {concept} in simple terms.")
])

filled_chat = chat_template.invoke({
    "domain": "machine learning",
    "concept": "gradient descent"
})
print("\nType 2 — Filled ChatPromptTemplate:")
print(filled_chat)   

# Pass to LLM
response2 = llm.invoke(filled_chat)
print("Answer:", response2.content)