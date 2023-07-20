import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import mols2grid

st.title("Checking served predictions")
st.markdown(
    """
    This page is for served predictions only.
    """
)
pred_file = st.file_uploader(
    "Upload a CSV file containing the predictions here",
    type="csv",
    key="predictions",
)

# save the df from the uploaded file in session state, so it persists
if pred_file is not None:
    st.session_state["df_pred"] = pd.read_csv(pred_file)

if "df_pred" in st.session_state:
    df_pred = st.session_state["df_pred"]
    # unique proteins, sorted alphabetically
    proteins = sorted(df_pred["Protein"].unique())

    # here, i won't know if a prediction is correct or not (new ligands)
    selected_protein = st.selectbox("Select a protein", proteins)
    df_pred_filtered = df_pred.query("Protein == @selected_protein")

    # sort by P (class 1)
    df_pred_filtered = df_pred_filtered.sort_values(by="P (class 1)", ascending=False)

    st.header(f"Predictions for {selected_protein}")

    raw_html = mols2grid.display(
        df_pred_filtered,
        smiles_col="Canon_SMILES",
        subset=["img", "Predicted value", "P (class 1)", "diff_p1_p0"],
        size=(225, 175),
        n_cols=3,
        tooltip=None,
        transform={
            "Predicted value": lambda x: f"Predicted class: {x}",
            "P (class 1)": lambda x: f"P (class 1): {x:.3f}",
            "diff_p1_p0": lambda x: f"VA interval: {x:.3f}",
        },
    )._repr_html_()
    components.html(raw_html, height=900, scrolling=True)
