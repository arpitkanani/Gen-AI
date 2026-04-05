from langchain_ollama import ChatOllama

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model=ChatOllama(model="llama3.2:1b")

chat_history=[
    SystemMessage(content="You are an expert AI assitance in every field of knowledge. Answer precisely and concisely."),

]

print("for chat quit write quit/exit or /bye")
while True:
    user_input=input("You: ")
    chat_history.append(HumanMessage(content=user_input)) # type: ignore
    if user_input.lower() in ["exit", "quit"] or user_input == "/bye":
        print("Exiting chat. see ya!")
        break

    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content)) # type: ignore
    print("AI: ", result.content)

print(chat_history)