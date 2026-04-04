from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    PromptTemplate,
)



llm = ChatOllama(model="llama3.2:1b")



template1 = PromptTemplate(
    template="Explain {topic} in simple terms for a {audience}.",
    input_variables=["topic", "audience"]
)


filled_prompt = template1.invoke({
    "topic": "neural networks",
    "audience": "10 year old kid"
})
print("\nType 1 — Filled PromptTemplate:")
print(filled_prompt)

# Pass to LLM
response1 = llm.invoke(filled_prompt)
print("Answer:", response1.content)

