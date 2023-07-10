import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import base64
from io import BytesIO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.title("Predictions (with calibration)")

def smi2img(smi):
    mol = Chem.MolFromSmiles(smi)
    img = Draw.MolToImage(mol, size=(300, 300))  # Change size as needed
    return img

df = pd.read_csv("cal_test_set_preds.csv")
df["Class"] = df["Class"].astype("int")
df["Predicted value"] = df["Predicted value"].astype("int")

df["Correct"] = df["Class"] == df["Predicted value"]

# unique proteins, sorted alphabetically
proteins = sorted(df["Protein"].unique())

selected_protein = st.selectbox("Select a protein", proteins)

df_filtered = df.query("Protein == @selected_protein")
correct_ligands = df_filtered.query("Correct == True")
incorrect_ligands = df_filtered.query("Correct == False")

st.header(f"Test set information for {selected_protein}")
# get a summary, starting with test set composition
col_a, col_b, col_c = st.columns(3)
col_a.metric(label=f"Total ligands for {selected_protein}", value=len(df_filtered))
col_b.metric(label="Actives", value=len(df_filtered.query("Class == 1")))
col_c.metric(label="Inactives", value=len(df_filtered.query("Class == 0")))

# make confusion matrix
cm = confusion_matrix(df_filtered["Class"], df_filtered["Predicted value"])

# display confusion matrix
st.subheader("Confusion matrix")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Inactive", "Active"])
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax, cmap="rocket")
st.pyplot(fig, use_container_width=True)

# sort by P (class 1)
correct_ligands = correct_ligands.sort_values(by="P (class 1)", ascending=False)
incorrect_ligands = incorrect_ligands.sort_values(by="P (class 1)", ascending=False)

st.header(f"Individual predictions for {selected_protein}")
col1, col2 = st.columns(2, gap="large")

col1.subheader("Correct predictions")
for i, row in correct_ligands.iterrows():
    col1.image(
        smi2img(row["Canon_SMILES"]),
        caption=f"True: {row['Class']}, Predicted: {row['Predicted value']}, P(1) = {row['P (class 1)']:0.3f}",
        use_column_width=True
    )

col2.subheader("Incorrect predictions")
for i, row in incorrect_ligands.iterrows():
    col2.image(
        smi2img(row["Canon_SMILES"]),
        caption=f"True: {row['Class']}, Predicted: {row['Predicted value']}, P(1) = {row['P (class 1)']:0.3f}",
        use_column_width=True
    )


# to center metrics
# https://discuss.streamlit.io/t/is-there-a-way-to-center-all-the-elements-in-st-metric/35136
css='''
[data-testid="metric-container"] {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] > div {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] label {
    width: fit-content;
    margin: auto;
}
'''
st.markdown(f'<style>{css}</style>',unsafe_allow_html=True)
