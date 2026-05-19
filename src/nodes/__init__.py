import json
import ast
import re
import concurrent.futures

from openai import OpenAI

from src.config import OPENAI_MODEL, TEMPERATURE
from src.state import AgentState

_SYSTEM_MSG = (
    "You are a distinguished organic chemistry professor at a top research university. "
    "Your explanations combine rigorous mechanistic depth with clarity, always connecting "
    "fundamental theory to real synthetic applications."
)

# ── Context search helpers ────────────────────────────────────────────────────

def _fetch_wikipedia(query: str) -> str:
    try:
        import wikipedia
        titles = wikipedia.search(query, results=3)
        parts = []
        for title in titles[:2]:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                parts.append(f"[Wikipedia – {title}]\n{page.summary[:1800]}")
            except Exception:
                pass
        return "\n\n".join(parts)
    except Exception:
        return ""


def _fetch_web(query: str) -> str:
    try:
        from langchain_tavily import TavilySearch
        tool = TavilySearch(max_results=3)
        results = tool.invoke(f"organic chemistry {query}")
        if isinstance(results, list):
            return "\n\n".join(
                f"[Web – {r.get('url', 'source')}]\n{r.get('content', '')[:900]}"
                for r in results[:3]
            )
    except Exception:
        pass
    return ""


def search_context(state: AgentState) -> dict:
    """Run Wikipedia and Tavily searches in parallel and merge context."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        wiki_f = pool.submit(_fetch_wikipedia, state["query"])
        web_f = pool.submit(_fetch_web, state["query"])
        wiki = wiki_f.result()
        web = web_f.result()

    parts = [p for p in (wiki, web) if p]
    return {"context": "\n\n---\n\n".join(parts) if parts else ""}


# ── LLM call ─────────────────────────────────────────────────────────────────

_PROMPT = """\
You are explaining organic chemistry to a Masters / PhD-level student in the
style of a classic textbook (Clayden, March, Carey & Sundberg). Chemists
learn from STRUCTURES and ARROWS, not paragraphs of prose. ChatGPT and
Gemini drown the reader in text — you must do the opposite.

REFERENCE CONTEXT (from Wikipedia and web — use this to enrich your answer):
{context}

USER QUESTION: {query}

═══════════════════════════════════════════════════════════════════════════
HARD RULE — REACTION DENSITY
═══════════════════════════════════════════════════════════════════════════
You MUST embed AT LEAST THREE <RXN> blocks (4–6 is ideal) inline with your
prose. Each <RXN> renders as a drawn reaction diagram.

Each <RXN> must show a COMPLETE transformation: real reactant(s) → real
product(s) — molecules a chemist could weigh out and a chemist could
isolate. Do NOT use <RXN> to show mechanistic arrows, single elementary
steps, or naked intermediates (acylium, carbocation, arenium, TS — the
SMILES for these do not parse and the diagram comes out broken).

Pick reactions that COMPLEMENT each other:
  1. Canonical example of the transformation (simplest real substrate)
  2. A scope variant on a different / more interesting substrate
  3. A regio- or stereoselective example
  4. A real industrial / pharma / total-synthesis application

For a molecule question, use <RXN> blocks to show key syntheses, named
reactions the molecule participates in, or characteristic transformations.

Keep prose between diagrams SHORT — 2–4 sentences max per segment. Use the
prose for arrow-pushing language ("the nucleophile attacks…", "the
arenium intermediate is stabilized by…"); let the diagrams carry the
structural information.

═══════════════════════════════════════════════════════════════════════════
STRUCTURE — pick reactions to ILLUSTRATE, not to enumerate every step
═══════════════════════════════════════════════════════════════════════════
Use these sections (skip any that don't apply). Each <RXN> must show a
COMPLETE transformation between stable, isolable molecules — NOT a single
mechanistic step with a naked cation, anion, radical, or transition state
on one side. Explain mechanism in prose with arrow-pushing language; let
the diagrams show real chemistry that actually goes in a flask.

## Overview
One short paragraph. Then a <RXN> of the simplest canonical example of the
transformation (real reactant → real product).

## Mechanism (prose-led)
Walk through the elementary steps in WORDS — "the nucleophile attacks the
electrophilic carbon, the leaving group departs in a concerted backside
manner." Do NOT make a <RXN> for each arrow. The diagrams already say what
goes in and what comes out.

## Stereochemistry & Regiochemistry
Show a stereodefined or regioselective example: e.g. <RXN> with (R)/(S)
or E/Z SMILES (`[C@H]`, `/C=C/`).

## Scope & Variants
Pick TWO genuinely different substrates — one electron-rich, one electron-
poor; or one cyclic, one acyclic; or a heterocyclic analogue. Show each as
a <RXN>. This is where you out-textbook ChatGPT — give SPECIFIC substrates
(anisole, indole, naphthalene, a steroid scaffold) rather than the bland
benzene/toluene defaults.

## Applications
One <RXN> of a real industrial, pharmaceutical, or natural-product synthesis
using this transformation. Name the target if you can.

═══════════════════════════════════════════════════════════════════════════
<RXN> BLOCK FORMAT
═══════════════════════════════════════════════════════════════════════════
Each diagram is a single JSON object wrapped in <RXN>…</RXN>:

<RXN>{{"smiles": "COc1ccccc1.CC(=O)Cl>>COc1ccc(C(C)=O)cc1", "reagents": ["AlCl3"], "conditions": ["DCM", "0 °C"], "caption": "Friedel–Crafts acylation of anisole — para-selective"}}</RXN>

Rules — read carefully, the renderer is strict:
• "smiles": reaction SMILES "reactants>>products" (REQUIRED, must contain ">>").
  Use "." to separate co-reactants or co-products.
• Every fragment MUST parse in RDKit. Test mentally before emitting.
• ❌ FORBIDDEN — these break the renderer and will display as red "invalid"
  tiles. DO NOT use:
    – Bare acylium / carbocations / oxocarbenium without valid SMILES
      (e.g. `CC(=O)[+]`, `[CH3+]` alone, `C[C+]=O` in a reaction block)
    – Standalone transition-state notation (`[‡]`, brackets, dotted bonds)
    – Arenium / Wheland intermediates as products
    – Tetrahedral intermediates with explicit charges
    – Bare radicals (`[CH3]` with a dot), nitrenes, carbenes
    – Pseudo-SMILES like "R-X", "Nu:", or anything with R/X/Nu letters
• ✓ ALLOWED — full neutral molecules, including charged species ONLY when
  they have a complete, valid SMILES (e.g. quaternary ammonium salts
  `[N+](C)(C)(C)C.[Cl-]`, ylides `[C-]([P+](c1ccccc1)(c1ccccc1)c1ccccc1)`).
• Include ONLY organic (carbon-based) species in "smiles".
  Inorganic acids (HNO3, H2SO4, HCl), bases (K2CO3, NaOH), metal catalysts
  (Pd, AlCl3, FeBr3), simple salts, solvents — these go in "reagents" or
  "conditions", NEVER in "smiles".
• "reagents": list of catalysts/reagents shown above the arrow (may be []).
• "conditions": list of solvent/temp/time shown below the arrow (may be []).
• "caption": short label (≤ 90 chars) describing what this diagram shows.
• The JSON must be valid — double quotes, no trailing commas, no line breaks
  inside the JSON object.

If you find yourself wanting to "show the acylium ion" or "show the
arenium intermediate," DON'T. Write about it in prose, and let the next
<RXN> show the next stable transformation.

═══════════════════════════════════════════════════════════════════════════
END BLOCK
═══════════════════════════════════════════════════════════════════════════
After all sections and <RXN> blocks, append EXACTLY this (nothing after):

<CHEMISTRY_DATA>
{{"follow_up": ["Q1?", "Q2?", "Q3?"]}}
</CHEMISTRY_DATA>

"follow_up" = exactly 3 thought-provoking questions a grad student would ask
next. Valid JSON, double quotes, no trailing commas.
"""


def stream_llm(query: str, context: str):
    """
    Generator that streams LLM response chunks for live display.
    Falls back to a single blocking call if streaming fails.
    """
    client = OpenAI()
    context_trimmed = (context or "No additional context available.")[:3500]
    messages = [
        {"role": "system", "content": _SYSTEM_MSG},
        {"role": "user", "content": _PROMPT.format(query=query, context=context_trimmed)},
    ]
    try:
        stream = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=TEMPERATURE,
            stream=True, messages=messages,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except Exception:
        response = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=TEMPERATURE, messages=messages,
        )
        yield response.choices[0].message.content.strip()


def call_llm(state: AgentState) -> dict:
    """Non-streaming version used by the LangGraph pipeline."""
    full = "".join(stream_llm(state["query"], state.get("context") or ""))
    return {"llm_response": full}


# ── Response parser ───────────────────────────────────────────────────────────

_DATA_RE = re.compile(
    r"<CHEMISTRY_DATA>\s*(.*?)\s*</CHEMISTRY_DATA>", re.DOTALL | re.IGNORECASE
)
_RXN_RE = re.compile(
    r"<RXN>\s*(\{.*?\})\s*</RXN>", re.DOTALL | re.IGNORECASE
)


def _load_json(blob: str):
    try:
        return json.loads(blob), None
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(blob), None
        except Exception as exc:
            return None, str(exc)


def parse_response(state: AgentState) -> dict:
    """
    Parse the LLM response into:
      • segments: ordered list of ('text', markdown_str) and ('rxn', dict) items
      • follow_up: 3 grad-student questions from the trailing CHEMISTRY_DATA block
      • reactions: flat list of all parsed reaction dicts (for convenience)
      • text: full prose with <RXN> blocks stripped (for fallback / re-display)
    """
    raw = state["llm_response"]

    # Strip the trailing CHEMISTRY_DATA block from the body before segmenting
    follow_up = None
    parse_error = None
    data_match = _DATA_RE.search(raw)
    if data_match:
        body = raw[: data_match.start()]
        data, err = _load_json(data_match.group(1).strip())
        if data:
            follow_up = data.get("follow_up") or None
        elif err:
            parse_error = f"follow-up JSON: {err}"
    else:
        body = raw
        parse_error = "LLM did not return a <CHEMISTRY_DATA> block."

    # Walk through the body, slicing on <RXN>...</RXN>
    segments: list[tuple[str, object]] = []
    reactions: list[dict] = []
    cursor = 0
    rxn_errors: list[str] = []

    for m in _RXN_RE.finditer(body):
        # Text before this reaction
        pre = body[cursor : m.start()].strip()
        if pre:
            segments.append(("text", pre))

        data, err = _load_json(m.group(1).strip())
        if data and data.get("smiles") and ">>" in str(data["smiles"]):
            rxn = {
                "smiles": data["smiles"],
                "reagents": data.get("reagents") or None,
                "conditions": data.get("conditions") or None,
                "caption": data.get("caption") or None,
            }
            segments.append(("rxn", rxn))
            reactions.append(rxn)
        elif err:
            rxn_errors.append(err)

        cursor = m.end()

    # Trailing text after the last reaction
    tail = body[cursor:].strip()
    if tail:
        segments.append(("text", tail))

    # If no segments at all, surface the raw body so the user still sees something
    if not segments:
        segments = [("text", body.strip())]

    full_text = "\n\n".join(s for kind, s in segments if kind == "text")

    error = parse_error
    if rxn_errors:
        joined = "; ".join(rxn_errors[:3])
        error = f"{error + ' | ' if error else ''}reaction JSON: {joined}"

    return {
        "text": full_text,
        "segments": segments,
        "reactions": reactions,
        "follow_up": follow_up,
        "error": error,
    }
