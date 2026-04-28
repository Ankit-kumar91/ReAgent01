# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ReAgent01 is a chemistry-focused AI agent built with LangGraph and LangChain. It accepts natural-language chemistry queries, retrieves structured answers (including SMILES strings) from an LLM, and visualizes molecules and reactions using RDKit. The Streamlit app (`streamlit_app.py`) is the intended user-facing interface.

## Commands

This project uses `uv` for dependency management (Python 3.13).

```bash
# Install dependencies
uv sync

# Run the Streamlit app
uv run streamlit run streamlit_app.py

# Run the main entry point
uv run python main.py

# Launch Jupyter for the document ingestion notebook
uv run jupyter notebook src/documents_ingestion/notebook_DI.ipynb
```

## Architecture

The `src/` package is organized around a LangGraph agent pattern:

- **`src/state/`** — Defines the shared `AgentState` (TypedDict or Pydantic model) passed between graph nodes.
- **`src/nodes/`** — Individual graph node functions (e.g., LLM call, tool invocation, SMILES parsing).
- **`src/graph_builder/`** — Assembles LangGraph `StateGraph`, wires nodes and conditional edges, and compiles the runnable.
- **`src/config/`** — Centralizes model names, temperature, and other runtime settings loaded from `.env`.
- **`src/vectorstore/`** — FAISS-backed vector store for RAG; handles document embedding and similarity search.
- **`src/documents_ingestion/`** — Preprocessing pipeline that loads, chunks, and embeds documents into the vector store.

### Chemistry domain conventions

- All molecule data is exchanged as **SMILES strings**. Reactions use reaction SMILES (`reactants>>products`).
- `query_llm()` (see notebook) prompts GPT-4o-mini to return a Python dict with `"text"` and `"smiles"` keys; parse responses with `json.loads` or `ast.literal_eval`.
- RDKit is used for rendering: `Draw.MolsToGridImage` for molecules, `Draw.ReactionToImage` for reactions. When drawing multiple molecules from a dot-separated SMILES string, split on commas first, then dots — dots inside aromatic ring notation must not be split.

## Environment Variables

Create a `.env` file at the project root (already gitignored):

```
OPENAI_API_KEY=...
LANGCHAIN_API_KEY=...
TAVILY_API_KEY=...
```

Load with `python-dotenv`; `src/config/` is the canonical place to expose these to the rest of the app.
