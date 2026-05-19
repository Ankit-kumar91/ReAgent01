"""
Beautiful reaction and molecule rendering using matplotlib + RDKit.

Reaction layout:
    [R1] + [R2]  ──reagents──►  [P1] + [P2]
                   conditions
"""
import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw

_MOL_PX = 210          # molecule image size (square)
_PLUS_PX = 52          # width of "+" gap between co-reactants
_ARROW_PX = 230        # width of arrow section
_PAD = 24              # outer horizontal padding
_LABEL_H = 26          # vertical space per reagent/condition line
_LABEL_PAD = 8         # gap between molecule row and label text
_DPI = 120


def _placeholder_array(smiles: str) -> np.ndarray:
    """A grey box with the broken SMILES — used when RDKit can't parse it."""
    img = Image.new("RGB", (_MOL_PX, _MOL_PX), color="#f6e7e7")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 13
        )
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle(
        [(2, 2), (_MOL_PX - 3, _MOL_PX - 3)],
        outline="#a83232", width=2,
    )
    draw.text(
        (_MOL_PX / 2, 26),
        "⚠ invalid SMILES",
        fill="#a83232", anchor="mm", font=font,
    )
    # Wrap the SMILES across multiple lines so long strings stay readable.
    snippet = smiles.strip()[:90]
    chunk = 18
    lines = [snippet[i : i + chunk] for i in range(0, len(snippet), chunk)]
    y = 60
    for line in lines[:8]:
        draw.text((_MOL_PX / 2, y), line, fill="#5a1414", anchor="mm", font=font)
        y += 18
    return np.asarray(img)


def _mol_to_array(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(_MOL_PX, _MOL_PX))
    return np.asarray(img)


def draw_reaction(
    reaction_smiles: str,
    reagents: Optional[list] = None,
    conditions: Optional[list] = None,
) -> tuple[Optional[Image.Image], list[str]]:
    """
    Return (image, failed_smiles).
      • image: PIL Image of the reaction, or None if the SMILES is structurally
        invalid (missing ">>" or empty on both sides).
      • failed_smiles: list of individual SMILES fragments RDKit could not parse.
        Unparseable fragments are still drawn as red "⚠ invalid SMILES" tiles so
        the diagram stays balanced and the caller can warn the user explicitly.
    """
    parts = reaction_smiles.split(">>")
    if len(parts) != 2:
        return None, [reaction_smiles]

    r_smi = [s for s in parts[0].split(".") if s.strip()]
    p_smi = [s for s in parts[1].split(".") if s.strip()]

    failed: list[str] = []
    r_arrs: list[np.ndarray] = []
    p_arrs: list[np.ndarray] = []
    for s in r_smi:
        arr = _mol_to_array(s)
        if arr is None:
            failed.append(s)
            r_arrs.append(_placeholder_array(s))
        else:
            r_arrs.append(arr)
    for s in p_smi:
        arr = _mol_to_array(s)
        if arr is None:
            failed.append(s)
            p_arrs.append(_placeholder_array(s))
        else:
            p_arrs.append(arr)

    if not r_arrs and not p_arrs:
        return None, failed

    n_r, n_p = len(r_arrs), len(p_arrs)

    # Vertical space for label text above / below the molecule row
    n_above = len(reagents) if reagents else 0
    n_below = len(conditions) if conditions else 0
    top_extra = max(n_above, 1) * _LABEL_H + _LABEL_PAD + _PAD
    bot_extra = max(n_below, 1) * _LABEL_H + _LABEL_PAD + _PAD

    total_w = (
        _PAD
        + n_r * _MOL_PX + max(0, n_r - 1) * _PLUS_PX
        + _ARROW_PX
        + n_p * _MOL_PX + max(0, n_p - 1) * _PLUS_PX
        + _PAD
    )
    total_h = top_extra + _MOL_PX + bot_extra

    fig = plt.figure(figsize=(total_w / _DPI, total_h / _DPI), dpi=_DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    mol_bot = bot_extra
    mol_top = bot_extra + _MOL_PX
    mol_mid = (mol_bot + mol_top) / 2.0

    # ── Reactants ──────────────────────────────────────────────────────────
    x = float(_PAD)
    for i, arr in enumerate(r_arrs):
        ax.imshow(
            arr,
            extent=[x, x + _MOL_PX, mol_bot, mol_top],
            origin="upper",
            aspect="auto",
            zorder=2,
        )
        x += _MOL_PX
        if i < n_r - 1:
            ax.text(
                x + _PLUS_PX / 2, mol_mid, "+",
                ha="center", va="center",
                fontsize=22, fontweight="bold", color="#2c2c2c",
                zorder=3,
            )
            x += _PLUS_PX

    # ── Arrow ──────────────────────────────────────────────────────────────
    ax0 = x + 14
    ax1 = x + _ARROW_PX - 14
    arrow_cx = (ax0 + ax1) / 2.0

    ax.annotate(
        "",
        xy=(ax1, mol_mid),
        xytext=(ax0, mol_mid),
        arrowprops=dict(
            arrowstyle="->",
            color="#1a1a1a",
            lw=2.5,
            mutation_scale=24,
        ),
        zorder=3,
    )

    if reagents:
        ax.text(
            arrow_cx,
            mol_top + _LABEL_PAD,
            "\n".join(reagents),
            ha="center", va="bottom",
            fontsize=9, color="#064e8c",
            style="italic",
            multialignment="center",
            zorder=3,
        )

    if conditions:
        ax.text(
            arrow_cx,
            mol_bot - _LABEL_PAD,
            "\n".join(conditions),
            ha="center", va="top",
            fontsize=9, color="#7a3810",
            style="italic",
            multialignment="center",
            zorder=3,
        )

    x += _ARROW_PX

    # ── Products ───────────────────────────────────────────────────────────
    for i, arr in enumerate(p_arrs):
        ax.imshow(
            arr,
            extent=[x, x + _MOL_PX, mol_bot, mol_top],
            origin="upper",
            aspect="auto",
            zorder=2,
        )
        x += _MOL_PX
        if i < n_p - 1:
            ax.text(
                x + _PLUS_PX / 2, mol_mid, "+",
                ha="center", va="center",
                fontsize=22, fontweight="bold", color="#2c2c2c",
                zorder=3,
            )
            x += _PLUS_PX

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches=None,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy(), failed


def draw_molecules(smiles_input) -> Optional[Image.Image]:
    """
    Draw a grid of molecules from a SMILES string or list of SMILES.
    Accepts comma-separated, dot-separated, or list input.
    """
    if isinstance(smiles_input, list):
        parts = [s.strip() for s in smiles_input if s.strip()]
    else:
        parts = [s.strip() for s in smiles_input.replace(",", ".").split(".") if s.strip()]

    mols = [Chem.MolFromSmiles(s) for s in parts]
    mols = [m for m in mols if m is not None]
    if not mols:
        return None

    return Draw.MolsToGridImage(
        mols,
        molsPerRow=min(len(mols), 4),
        subImgSize=(280, 280),
    )
