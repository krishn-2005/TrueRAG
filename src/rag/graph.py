from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from src.rag.nodes import (
    retrieve,
    generate,
    check_context,
    fallback
)

class State(TypedDict):

    question: str

    context: str

    sources: list[dict]

    answer: str


graph = StateGraph(State)

graph.add_node("retrieve", retrieve)

graph.add_node("generate", generate)

graph.add_node("fallback", fallback)

graph.add_edge(START, "retrieve")

graph.add_conditional_edges(

    "retrieve",

    check_context,

    {

        "has_context": "generate",

        "no_context": "fallback"

    }

)

graph.add_edge("generate", END)

graph.add_edge("fallback", END)

app = graph.compile()