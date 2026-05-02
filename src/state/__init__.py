from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    query: str
    llm_response: str
    text: str
    smiles: Any          # str (single/dot-joined/reaction) or list[str]
    error: Optional[str]
