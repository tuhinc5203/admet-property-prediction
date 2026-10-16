"""
Interactive Streamlit demo for the ADMET property
prediction project. Paste a SMILES string, get predictions for all 5
trained ADMET properties with plain-language interpretation.

Reuses the exact featurization pipeline from notebooks/Model_Improvement.ipynb
(largest-fragment salt stripping, 2048-bit Morgan fingerprint + 6 RDKit
descriptors) so predictions match what the notebooks report. Models are
the final tuned-XGBoost `best_estimator_` objects from Experiment 3,
saved to models/*.pkl by that notebook's last cell.
"""

import os
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.DataStructs import ConvertToNumpyArray

MODEL_DIR = "models"
PUBCHEM_HEADERS = {"User-Agent": "admet-property-prediction-demo"}

EXAMPLE_MOLECULES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Diazepam": "CN1C(=O)CN=C(C2=CC=CC=C2)C2=CC(Cl)=CC=C12",
    "Loperamide": "CCC(CC1=CC=CC=C1)(C1=CC=CC=C1)C(=O)N(C)CCC(O)(C1=CC=CC=C1)C1=CC=CC=C1",
}

# Same 5 properties, same order as the README's results table.
PROPERTIES = ["Solubility", "BBB", "hERG", "CYP3A4", "Clearance"]

MODEL_FILES = {
    "Solubility": "solubility_xgb.pkl",
    "BBB": "bbb_xgb.pkl",
    "hERG": "herg_xgb.pkl",
    "CYP3A4": "cyp3a4_xgb.pkl",
    "Clearance": "clearance_xgb.pkl",
}


# ---------------------------------------------------------------------------
# Featurization -- unchanged from notebooks/Model_Improvement.ipynb, adapted to a
# single molecule instead of a DataFrame of many.
# ---------------------------------------------------------------------------

def largest_fragment(mol):
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def compute_morgan_fp(mol, radius=2, n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr


def featurize_one(mol, radius=2, n_bits=2048):
    fp = compute_morgan_fp(mol, radius=radius, n_bits=n_bits)
    descriptors = {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
    }
    fp_cols = [f"fp_{i}" for i in range(n_bits)]
    row = {**dict(zip(fp_cols, fp)), **descriptors}
    return pd.DataFrame([row]), descriptors


# ---------------------------------------------------------------------------
# PubChem name lookup -- lets users search by molecule name when they don't
# have a SMILES string handy, without requiring one (manual paste still works).
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def pubchem_search_names(query, limit=8):
    query = query.strip()
    if len(query) < 2:
        return []
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound/{quote(query)}/json"
    try:
        resp = requests.get(url, params={"limit": limit}, headers=PUBCHEM_HEADERS, timeout=5)
        resp.raise_for_status()
        return resp.json().get("dictionary_terms", {}).get("compound", [])
    except requests.RequestException:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def pubchem_name_to_smiles(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/property/SMILES/JSON"
    try:
        resp = requests.get(url, headers=PUBCHEM_HEADERS, timeout=5)
        if not resp.ok:
            return None
        properties = resp.json().get("PropertyTable", {}).get("Properties", [])
        return properties[0]["SMILES"] if properties else None
    except (requests.RequestException, KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    models = {}
    missing = []
    for prop, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[prop] = joblib.load(path)
        else:
            missing.append(path)
    return models, missing


# ---------------------------------------------------------------------------
# Plain-language interpretation
# ---------------------------------------------------------------------------

def interpret_solubility(logs):
    # Common LogS (log mol/L) solubility convention (e.g. Delaney/ESOL bins).
    if logs > -1:
        return "Highly soluble"
    if logs > -2:
        return "Soluble"
    if logs > -3:
        return "Moderately soluble"
    if logs > -4:
        return "Poorly soluble"
    return "Very poorly soluble"


def interpret_binary(prob, positive_label, negative_label):
    return positive_label if prob >= 0.5 else negative_label


def interpret_clearance(value):
    # Dataset's own observed range is ~3-150 uL/min/1e6 cells (Clearance_Hepatocyte_AZ).
    # Heuristic tertiles relative to that range, not a clinical cutoff.
    if value < 15:
        return "Low (relative to this dataset's typical 3-150 range)"
    if value < 50:
        return "Moderate (relative to this dataset's typical 3-150 range)"
    return "High (relative to this dataset's typical 3-150 range)"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ADMET Property Predictor", layout="centered")

# Minimal styling on top of the theme in .streamlit/config.toml.
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 3rem; max-width: 760px;}
    h1 {font-weight: 600; letter-spacing: -0.01em;}
    h2, h3 {font-weight: 500;}
    .stButton > button {border-radius: 4px; border: 1px solid #CFCABE; background: transparent;}
    .stButton > button:hover {border-color: #3D5A6C; color: #3D5A6C;}
    .stButton > button[kind="primary"] {background: #3D5A6C; color: #FAF8F4; border-color: #3D5A6C;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ADMET Property Predictor")
st.caption(
    "Predict five ADMET properties from a molecule's structure: aqueous solubility, "
    "blood-brain barrier (BBB) permeability, hERG cardiotoxicity risk, CYP3A4 "
    "metabolic inhibition, and hepatocyte clearance."
)

models, missing = load_models()
if missing:
    st.error(
        "Model file(s) not found: " + ", ".join(missing) + ". "
        "Run notebooks/Model_Improvement.ipynb's last cell to generate them."
    )
    st.stop()

if "smiles_input" not in st.session_state:
    st.session_state.smiles_input = EXAMPLE_MOLECULES["Aspirin"]

st.markdown("**Examples**")
example_cols = st.columns(len(EXAMPLE_MOLECULES))
for col, (name, smi) in zip(example_cols, EXAMPLE_MOLECULES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.smiles_input = smi

with st.expander("Search PubChem by molecule name"):
    st.caption("No SMILES string handy? Look a molecule up by name and select a match.")
    name_query = st.text_input("Molecule name", placeholder="e.g. aspirin", key="name_query")

    if name_query.strip():
        matches = pubchem_search_names(name_query)
        if matches:
            match_cols = st.columns(4)
            for i, match_name in enumerate(matches):
                if match_cols[i % 4].button(match_name, key=f"pubchem_match_{i}_{match_name}", use_container_width=True):
                    smi = pubchem_name_to_smiles(match_name)
                    if smi:
                        st.session_state.smiles_input = smi
                    else:
                        st.error(f"Couldn't fetch a SMILES string for '{match_name}' from PubChem.")
        else:
            st.caption("No PubChem matches yet. Keep typing, or paste a SMILES string below.")

smiles = st.text_input("SMILES string", key="smiles_input")
predict_clicked = st.button("Predict", type="primary")

if predict_clicked:
    mol_raw = Chem.MolFromSmiles(smiles)
    if mol_raw is None:
        st.error("Couldn't parse that SMILES string. Check the syntax and try again.")
        st.stop()

    mol = largest_fragment(mol_raw)
    X, descriptors = featurize_one(mol)

    st.divider()
    col_img, col_desc = st.columns([1, 1.2])
    with col_img:
        img = Draw.MolToImage(mol, size=(300, 300), fitImage=True)
        st.image(img, caption="Largest fragment used for prediction")
    with col_desc:
        st.markdown("**Computed descriptors**")
        desc_df = pd.DataFrame(
            {"Value": [f"{v:.2f}" for v in descriptors.values()]},
            index=descriptors.keys(),
        )
        st.table(desc_df)

    st.subheader("Predictions")

    sol_pred = models["Solubility"].predict(X)[0]
    sol_label = interpret_solubility(sol_pred)

    bbb_prob = models["BBB"].predict_proba(X)[0, 1]
    bbb_label = interpret_binary(bbb_prob, "Permeable", "Not permeable")

    herg_prob = models["hERG"].predict_proba(X)[0, 1]
    herg_label = interpret_binary(herg_prob, "Potential blocker (caution)", "Low risk")

    cyp_prob = models["CYP3A4"].predict_proba(X)[0, 1]
    cyp_label = interpret_binary(cyp_prob, "Predicted inhibitor", "Predicted non-inhibitor")

    clr_pred = models["Clearance"].predict(X)[0]
    clr_label = interpret_clearance(clr_pred)

    rows = [
        ("Solubility (LogS)", f"{sol_pred:.2f}", sol_label),
        ("BBB Permeability", f"{bbb_prob:.1%} probability", bbb_label),
        ("hERG Blockade", f"{herg_prob:.1%} probability", herg_label),
        ("CYP3A4 Inhibition", f"{cyp_prob:.1%} probability", cyp_label),
        ("Clearance (uL/min/1e6 cells)", f"{clr_pred:.1f}", clr_label),
    ]
    results_df = pd.DataFrame(rows, columns=["Property", "Model output", "Interpretation"])
    st.table(results_df.set_index("Property"))

    st.caption(
        "Educational project, not for clinical or regulatory use. Model performance "
        "and known limitations (e.g. failure modes for charged or atypical molecules) "
        "are documented in the README and docs/NOTES.md."
    )

st.divider()
st.caption(
    "Tuned XGBoost model per property, trained on scaffold splits. Source: "
    "[admet-property-prediction](https://github.com/tuhinc5203/admet-property-prediction)."
)
