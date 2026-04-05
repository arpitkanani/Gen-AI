from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal

model = ChatOllama(model="llama3.2:3b")
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment of the feedback"
    )

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment classifier. Classify feedback as positive, negative or neutral."),
    ("human", "Classify this feedback: {feedback}")
])

classifier_chain = classifier_prompt | model.with_structured_output(Feedback)


prompt_positive = ChatPromptTemplate.from_messages([
    ("system", "You write warm, grateful responses to positive feedback."),
    ("human", "Write a response to this positive feedback: {feedback}")
])

prompt_negative = ChatPromptTemplate.from_messages([
    ("system", "You write empathetic, solution-focused responses to negative feedback."),
    ("human", "Write a response to this negative feedback: {feedback}")
])


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt_positive | model | parser), # type: ignore
    (lambda x: x.sentiment == "negative", prompt_negative | model | parser), # type: ignore
    RunnableLambda(lambda x: "Neutral feedback — no response needed.")
)



full_chain = (
    RunnablePassthrough.assign(sentiment_obj=classifier_chain)
    | RunnableLambda(lambda x: type(
        "Combined", (), {
            "sentiment": x["sentiment_obj"].sentiment, # type: ignore
            "feedback": x["feedback"] # type: ignore
        }
    )())
    | branch_chain
)

feedbacks = [
    "I love the new update, it's fantastic!",
    "This is terrible, nothing works properly.",
    "It's okay I guess, nothing special."
]

for fb in feedbacks:
    print(f"\nFeedback: {fb}")
    result = full_chain.invoke({"feedback": fb})
    print(f"Response: {result}")