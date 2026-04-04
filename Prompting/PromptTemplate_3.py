
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
# {chat_history} will be replaced by actual message objects
history_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Simulate a conversation history
history = [
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is a high level programming language known for simplicity."),
    HumanMessage(content="What is it used for?"),
    AIMessage(content="It is used for web development, data science, AI, and automation."),
]

# Fill template with history + new question
filled_history = history_template.invoke({
    "chat_history": history,
    "question": "Which of those fields pays the most?"
})
print("\n📌 Type 3 — With conversation history:")
response3 = llm.invoke(filled_history)
print("🤖 Answer:", response3.content)
