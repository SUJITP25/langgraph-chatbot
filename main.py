from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

print(llm)




class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    message = state["messages"]
    response = llm.invoke(message)

checkpoint = MemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

graph = graph.compile(checkpointer=checkpoint)