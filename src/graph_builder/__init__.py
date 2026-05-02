from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.nodes import call_llm, parse_response


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("call_llm", call_llm)
    g.add_node("parse_response", parse_response)
    g.add_edge(START, "call_llm")
    g.add_edge("call_llm", "parse_response")
    g.add_edge("parse_response", END)
    return g.compile()


app = build_graph()
