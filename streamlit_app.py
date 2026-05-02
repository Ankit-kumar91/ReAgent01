import ast

import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions

from src.graph_builder import app


# ── molecule / reaction rendering ────────────────────────────────────────────

def _parse_smiles_field(smiles):
    """Normalise the smiles field into a flat list of SMILES strings."""
    if smiles is None:
        return []
    if isinstance(smiles, str) and smiles.startswith("[") and smiles.endswith("]"):
        try:
            smiles = ast.literal_eval(smiles)
        except Exception:
            pass
    if isinstance(smiles, list):
        return [s.strip() for s in smiles if s and s.strip()]
    return [smiles.strip()]


def render_smiles(smiles):
    """Return a PIL Image for the given SMILES value, or None on failure."""
    parts = _parse_smiles_field(smiles)
    if not parts:
        return None

    images = []
    mol_batch = []

    def flush_mol_batch():
        if mol_batch:
            img = Draw.MolsToGridImage(
                mol_batch,
                molsPerRow=min(len(mol_batch), 4),
                subImgSize=(250, 250),
            )
            images.append(img)
            mol_batch.clear()

    for smi in parts:
        if ">>" in smi:
            flush_mol_batch()
            rxn = rdChemReactions.ReactionFromSmarts(smi, useSmiles=True)
            if rxn:
                images.append(Draw.ReactionToImage(rxn, subImgSize=(280, 200)))
        elif "," in smi:
            # comma-separated molecules inside a single string
            flush_mol_batch()
            sub_mols = [Chem.MolFromSmiles(p.strip()) for p in smi.split(",")]
            sub_mols = [m for m in sub_mols if m]
            if sub_mols:
                images.append(
                    Draw.MolsToGridImage(
                        sub_mols,
                        molsPerRow=min(len(sub_mols), 4),
                        subImgSize=(250, 250),
                    )
                )
        else:
            # may be a single molecule or a dot-separated mixture (not a reaction)
            sub_parts = smi.split(".")
            sub_mols = [Chem.MolFromSmiles(p.strip()) for p in sub_parts]
            sub_mols = [m for m in sub_mols if m]
            mol_batch.extend(sub_mols)

    flush_mol_batch()
    return images if images else None


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="ReAgent", page_icon="⚗️", layout="centered")
st.title("⚗️ ReAgent — Chemistry AI Assistant")
st.caption("Ask any organic chemistry question and get a visual answer.")

query = st.text_input(
    "Your chemistry query",
    placeholder="e.g. Show me the Diels-Alder reaction with butadiene and ethylene",
)

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Thinking…"):
        result = app.invoke(
            {
                "query": query,
                "llm_response": "",
                "text": "",
                "smiles": None,
                "error": None,
            }
        )

    if result.get("error"):
        st.error(f"Parse error: {result['error']}")
        with st.expander("Raw LLM response"):
            st.code(result.get("llm_response", ""), language="text")
    else:
        st.subheader("Explanation")
        st.write(result["text"])

        smiles = result.get("smiles")
        if smiles:
            st.subheader("Structure / Reaction")
            st.code(str(smiles), language="text")

            images = render_smiles(smiles)
            if images:
                for img in images:
                    st.image(img)
            else:
                st.warning("Could not render the SMILES structure.")
