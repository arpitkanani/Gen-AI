from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import HumanMessage, AIMessage

# Chat model (industry standard from now on)
llm = ChatOllama(model="llama3.2:1b")
print("✅ LLM ready!")

# from_messages takes a list of (role, content) tuples
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} teacher. Always give examples."),
    ("human", "Explain {concept} in simple terms.")
])

# Fill variables
filled_chat = chat_template.invoke({
    "domain": "machine learning",
    "concept": "gradient descent"
})
print("\n📌 Type 2 — Filled ChatPromptTemplate:")
print(filled_chat)   # notice it's a list of messages now

# Pass to LLM
response2 = llm.invoke(filled_chat)
print("🤖 Answer:", response2.content)