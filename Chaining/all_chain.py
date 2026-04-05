from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables import RunnableParallel


llm = ChatOllama(model="llama3.2:3b")
print("✅ LLM ready!")

prompt1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely."),
    ("human", "Explain {topic} in one sentence.")
])

parser = StrOutputParser()

chain1 = prompt1 | llm | parser

response1 = chain1.invoke({"topic": "vector databases"})
print("\nLevel 1 — Basic chain:")
print(response1)  # plain string, no .content needed ✅

prompt_explain = ChatPromptTemplate.from_messages([
    ("system", "You are a teacher. Be concise."),
    ("human", "Explain {concept} in 2 sentences.")
])

prompt_simplify = ChatPromptTemplate.from_messages([
    ("system", "You simplify technical text for beginners."),
    ("human", "Simplify this further: {explanation}")
])

chain_explain = prompt_explain | llm | parser
chain_simplify = prompt_simplify | llm | parser

sequential_chain = (
    chain_explain
    | RunnableLambda(lambda x: {"explanation": x})
    | chain_simplify
)

response2 = sequential_chain.invoke({"concept": "transformer attention"})
print("\nLevel 2 — Sequential chain:")
print(response2)


prompt_pros = ChatPromptTemplate.from_messages([
    ("system", "List only advantages. Be brief."),
    ("human", "Advantages of {technology}?")
])

prompt_cons = ChatPromptTemplate.from_messages([
    ("system", "List only disadvantages. Be brief."),
    ("human", "Disadvantages of {technology}?")
])

parallel_chain = RunnableParallel({
    "pros": prompt_pros | llm | parser,
    "cons": prompt_cons | llm | parser
})

response3 = parallel_chain.invoke({"technology": "RAG"})
print("\nLevel 3 — Parallel chain:")
print("PROS:", response3["pros"])
print("CONS:", response3["cons"])


def word_count_check(text):
    word_count = len(text.split())
    return f"[{word_count} words] {text}"

prompt4 = ChatPromptTemplate.from_messages([
    ("system", "Answer very concisely."),
    ("human", "What is {topic}?")
])

# RunnableLambda wraps any python function into a chain step
chain4 = prompt4 | llm | parser | RunnableLambda(word_count_check)

response4 = chain4.invoke({"topic": "embeddings"})
print("\nLevel 4 — Chain with custom function:")
print(response4)