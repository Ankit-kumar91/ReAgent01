import json
import ast

from openai import OpenAI

from src.config import OPENAI_MODEL, TEMPERATURE
from src.state import AgentState

_SYSTEM_MSG = "You are a senior organic chemist and synthesis planner."

_PROMPT = """\
You are an expert organic chemist specializing in synthesis design and reaction mechanisms.
Analyze the user's chemistry question and provide a scientific explanation plus valid SMILES.

Rules:
1. If the query is about a *single molecule* or *multiple molecules*:
   - Return canonical SMILES for each, comma-separated, as a single string.
2. If the query is about a *reaction*:
   - Return ONE reaction SMILES: reactants>>products  (use dots to join multiple reactants/products).

Output format — valid Python dict parsable with ast.literal_eval, no markdown:
  {{"text": "<concise explanation>", "smiles": "<smiles string>"}}

Examples:
  Molecules: {{"text": "Ethanol and acetone are common solvents.", "smiles": "CCO,CC(=O)C"}}
  Reaction:  {{"text": "Suzuki coupling of bromobenzene with phenylboronic acid.", "smiles": "Brc1ccccc1.B(O)(O)c1ccccc1>>c1ccc(-c2ccccc2)cc1"}}

User query: {query}
"""


def call_llm(state: AgentState) -> dict:
    client = OpenAI()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": _SYSTEM_MSG},
            {"role": "user", "content": _PROMPT.format(query=state["query"])},
        ],
    )
    return {"llm_response": response.choices[0].message.content.strip()}


def parse_response(state: AgentState) -> dict:
    raw = state["llm_response"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(raw)
        except Exception as exc:
            return {"text": raw, "smiles": None, "error": str(exc)}

    return {
        "text": data.get("text", ""),
        "smiles": data.get("smiles", ""),
        "error": None,
    }
