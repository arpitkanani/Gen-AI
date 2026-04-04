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

# ─────────────────────────────────────────
# TYPE 1 — PromptTemplate (simple string)
# ─────────────────────────────────────────
# Define template with {variables} in curly braces
template1 = PromptTemplate(
    template="Explain {topic} in simple terms for a {audience}.",
    input_variables=["topic", "audience"]
)

# Fill variables → gives you a filled string
filled_prompt = template1.invoke({
    "topic": "neural networks",
    "audience": "10 year old kid"
})
print("\n📌 Type 1 — Filled PromptTemplate:")
print(filled_prompt)

# Pass to LLM
response1 = llm.invoke(filled_prompt)
print("🤖 Answer:", response1.content)

