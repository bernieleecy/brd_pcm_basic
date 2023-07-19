import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mols2grid

from Start_here import read_csv

st.title("Checking served predictions")
st.markdown(
    """
    This page is for served predictions only.
    """)

# load in data for test set, hard coded during testing

file = st.file_uploader("Upload a CSV file containing the predictions here",
                        type="csv",)

if file is not None:
    df = read_csv(file)

    # unique proteins, sorted alphabetically
    proteins = sorted(df["Protein"].unique())

    # here, i won't know if a prediction is correct or not (new ligands)
    selected_protein = st.selectbox("Select a protein", proteins)
    df_filtered = df.query("Protein == @selected_protein")

    # sort by P (class 1)
    df_filtered = df_filtered.sort_values(by="P (class 1)", ascending=False)

    st.header(f"Predictions for {selected_protein}")

    raw_html = mols2grid.display(
        df_filtered,
        smiles_col="Canon_SMILES",
        subset=["img", "Predicted value", "P (class 1)", "diff_p1_p0"],
        size=(225,175),
        n_cols=3,
        tooltip=None,
        transform={"Predicted value": lambda x: f"Predicted class: {x}",
                   "P (class 1)": lambda x: f"P (class 1): {x:.3f}",
                   "diff_p1_p0": lambda x: f"VA interval: {x:.3f}"},
    )._repr_html_()
    components.html(raw_html, height=900, scrolling=True)
