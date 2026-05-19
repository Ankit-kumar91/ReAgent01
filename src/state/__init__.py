from typing import TypedDict, Optional, Any


class AgentState(TypedDict, total=False):
    query: str
    context: str                   # gathered from Wikipedia + web search
    llm_response: str
    text: str                      # prose with <RXN> blocks stripped
    segments: list                 # ordered [('text', md), ('rxn', dict), ...]
    reactions: list                # flat list of parsed reaction dicts
    follow_up: Optional[list]      # 3 follow-up questions for the student
    error: Optional[str]
