from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Connect to your local Ollama model
# This does NOT download anything — uses already pulled llama3.2:1b
llm = OllamaLLM(model="llama3.2:1b")
print("✅ LLM connected!")

# ─────────────────────────────────────────
# LEVEL 1 — Raw prompt (simplest)
# just send a plain string directly
# ─────────────────────────────────────────
response = llm.invoke("What is machine learning in one sentence?")
print("\n📌 Level 1 — Raw prompt:")
print(response)

# ─────────────────────────────────────────
# LEVEL 2 — Structured messages
# system sets role, human asks question
# ─────────────────────────────────────────
messages = [
    SystemMessage(content="You are an expert NLP engineer. Give short, precise answers."),
    HumanMessage(content="What is machine learning in one sentence?")
]
response2 = llm.invoke(messages)
print("\n📌 Level 2 — With System message:")
print(response2)

# ─────────────────────────────────────────
# LEVEL 3 — Conversation history
# showing LLM previous messages so it has context
# ─────────────────────────────────────────
messages2 = [
    SystemMessage(content="You are an expert NLP engineer. Give short, precise answers."),
    HumanMessage(content="What is machine learning?"),
    AIMessage(content="Machine learning is a method where computers learn patterns from data to make predictions."),
    HumanMessage(content="Can you give me a real world example of that?")
]
response3 = llm.invoke(messages2)
print("\n📌 Level 3 — With conversation history:")
print(response3)