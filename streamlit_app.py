import re

import streamlit as st

from src.nodes import search_context, parse_response, stream_llm
from src.rendering import draw_reaction

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ReAgent — Chemistry AI",
    page_icon="⚗️",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "auto_ask" not in st.session_state:
    st.session_state.auto_ask = False

# ── Header ────────────────────────────────────────────────────────────────────

st.title("⚗️ ReAgent — Organic Chemistry AI")
st.caption("Textbook-style explanations with multiple reaction diagrams · Wikipedia + Web search")
st.divider()

# ── Input row ─────────────────────────────────────────────────────────────────

q_col, btn_col = st.columns([7, 1])
with q_col:
    st.text_input(
        "query",
        placeholder="e.g. Explain aromatic electrophilic substitution with example",
        key="query_box",
        label_visibility="collapsed",
    )
with btn_col:
    ask_clicked = st.button("Ask ⚗️", type="primary", use_container_width=True)

# ── Determine whether to run ──────────────────────────────────────────────────

run_query: str | None = None
current_query = st.session_state.get("query_box", "").strip()

if ask_clicked and current_query:
    run_query = current_query
    st.session_state.result = None
elif st.session_state.auto_ask and current_query:
    run_query = current_query
    st.session_state.auto_ask = False

# ── Helpers ───────────────────────────────────────────────────────────────────

# During streaming we hide the <RXN>{...}</RXN> blocks and the trailing
# <CHEMISTRY_DATA>…</CHEMISTRY_DATA> block, replacing each <RXN> with a
# textbook-style "Figure N" placeholder so the prose still reads naturally.
_RXN_STREAM_RE = re.compile(r"<RXN>.*?</RXN>", re.DOTALL | re.IGNORECASE)
_RXN_OPEN_RE = re.compile(r"<RXN>", re.IGNORECASE)
_DATA_OPEN_RE = re.compile(r"<CHEMISTRY_DATA>", re.IGNORECASE)


def _streaming_view(text: str) -> str:
    """Hide raw machine-readable blocks while the model is still writing."""
    n = 0

    def _replace(_m):
        nonlocal n
        n += 1
        return f"\n\n_📐 Figure {n} — rendering…_\n\n"

    text = _RXN_STREAM_RE.sub(_replace, text)
    # Drop any unclosed <RXN> tag at the end (still streaming)
    text = _RXN_OPEN_RE.split(text)[0]
    # Hide the trailing CHEMISTRY_DATA block (closed or still streaming)
    text = _DATA_OPEN_RE.split(text)[0]
    return text


def _render_reaction(rxn: dict, idx: int, container=None):
    c = container or st
    smiles = rxn.get("smiles")
    if not smiles:
        return
    result = draw_reaction(
        smiles,
        reagents=rxn.get("reagents"),
        conditions=rxn.get("conditions"),
    )
    img, failed = result if isinstance(result, tuple) else (result, [])

    if img:
        caption = rxn.get("caption") or ""
        c.image(img, caption=f"Figure {idx}. {caption}" if caption else f"Figure {idx}")
    else:
        c.warning(f"Could not render reaction {idx}: `{smiles}`")

    if failed:
        bad = ", ".join(f"`{s}`" for s in failed)
        c.warning(
            f"Figure {idx}: RDKit could not parse {len(failed)} fragment(s): {bad}. "
            "Likely an intermediate (cation/anion/TS) — the prompt now asks the model "
            "to use only stable molecules. Try re-running the query."
        )

    with c.expander(f"Figure {idx} — SMILES details"):
        c.code(smiles, language="text")
        if rxn.get("reagents"):
            c.markdown(f"**Reagents:** {', '.join(rxn['reagents'])}")
        if rxn.get("conditions"):
            c.markdown(f"**Conditions:** {', '.join(rxn['conditions'])}")


def _render_segments(segments):
    """Render the textbook flow: alternating markdown and reaction figures."""
    fig_idx = 0
    for kind, payload in segments:
        if kind == "text":
            st.markdown(payload)
        elif kind == "rxn":
            fig_idx += 1
            _render_reaction(payload, fig_idx)


def _render_followup(result):
    questions = result.get("follow_up")
    if not questions:
        return
    st.divider()
    st.subheader("Explore further")
    st.caption("Click any question to ask it next.")
    cols = st.columns(len(questions))
    for i, q in enumerate(questions):
        with cols[i]:
            if st.button(q, key=f"fq_{i}", use_container_width=True):
                st.session_state.query_box = q
                st.session_state.result = None
                st.session_state.auto_ask = True
                st.rerun()


# ── Run agent with streaming ──────────────────────────────────────────────────

if run_query:
    # Step 1 — search (fast, shown as a status chip)
    with st.status("Searching Wikipedia and web…", expanded=False) as search_status:
        ctx_text = search_context({"query": run_query}).get("context", "")
        search_status.update(
            label="Sources found ✓" if ctx_text else "Using model knowledge ✓",
            state="complete",
        )

    # Step 2 — stream prose into a single placeholder.
    # We render the fully interleaved textbook flow after streaming finishes,
    # so reaction images appear in the right place rather than at the bottom.
    stream_box = st.empty()
    accumulated: list[str] = []
    char_buffer = 0
    REFRESH_EVERY = 24  # update UI every ~24 characters to reduce redraws

    for chunk in stream_llm(run_query, ctx_text):
        accumulated.append(chunk)
        char_buffer += len(chunk)
        if char_buffer >= REFRESH_EVERY:
            partial = _streaming_view("".join(accumulated))
            stream_box.markdown(partial + " ▌")
            char_buffer = 0

    full_response = "".join(accumulated)

    # Step 3 — parse into segments and re-render as textbook flow
    result = parse_response({"llm_response": full_response})
    st.session_state.result = result

    stream_box.empty()
    _render_segments(result["segments"])

    if result.get("error"):
        st.warning(f"Parsing note: {result['error']}")

    _render_followup(result)

# ── Re-display stored result (after follow-up rerun) ─────────────────────────

elif st.session_state.result:
    result = st.session_state.result
    _render_segments(result["segments"])
    if result.get("error"):
        st.warning(f"Parsing note: {result['error']}")
    _render_followup(result)
