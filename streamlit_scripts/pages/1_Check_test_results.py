import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import base64
from io import BytesIO
import mols2grid

st.title("Checking test set results")
st.markdown(
    """
    This page is for test set results only. If you made predictions on unseen
    data, please proceed to the next page (check served predictions).
    """)

# load in data for test set, hard coded during testing
file = st.file_uploader("Upload a CSV file containing test set results here", type="csv")

if file is not None:
    df = pd.read_csv(file)

    # unique proteins, sorted alphabetically
    proteins = sorted(df["Protein"].unique())

    selected_protein = st.selectbox("Select a protein", proteins)
    df_filtered = df.query("Protein == @selected_protein")

    # sort by P (class 1)
    df_filtered = df_filtered.sort_values(by="P (class 1)", ascending=False)

    st.header(f"Predictions for {selected_protein}")

    raw_html = mols2grid.display(
        df_filtered,
        smiles_col="Canon_SMILES",
        subset=["img", "Class", "Predicted value", "P (class 1)", "diff_p1_p0"],
        size=(225,175),
        n_cols=3,
        tooltip=["Closest Train SMILES", "Tanimoto Similarity"],
        transform={"Class": lambda x: f"True class: {x}",
                   "Predicted value": lambda x: f"Predicted class: {x}",
                   "P (class 1)": lambda x: f"P (class 1): {x:.3f}",
                   "diff_p1_p0": lambda x: f"VA interval: {x:.3f}",
                   "Tanimoto Similarity": lambda x: f"{x:.3f}"},
    )._repr_html_()
    components.html(raw_html, height=900, scrolling=True)
