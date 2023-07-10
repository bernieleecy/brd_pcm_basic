import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import base64
from io import BytesIO

st.title("Predictions (with calibration)")

def smi2img(smi):
    mol = Chem.MolFromSmiles(smi)
    img = Draw.MolToImage(mol, size=(300,300))
    return img

# load in data for test set
df = pd.read_csv("cal_test_set_preds_detailed.csv")
# load in data for train set
train_X = pd.read_parquet("../data/X_train.parquet", columns=["Canon_SMILES",
                                                               "Protein"])
train_y = pd.read_parquet("../data/y_train.parquet")
train_df = pd.concat([train_X, train_y], axis=1)

# unique proteins, sorted alphabetically
proteins = sorted(df["Protein"].unique())

selected_protein = st.selectbox("Select a protein", proteins)
pred_type = st.checkbox("Show misclassified instances only", value=True)

df_filtered = df.query("Protein == @selected_protein")
if pred_type:
    df_filtered = df_filtered.query("Correct == False")

# sort by P (class 1)
df_filtered = df_filtered.sort_values(by="P (class 1)", ascending=False)

st.header(f"Individual predictions for {selected_protein}")
col1, col2 = st.columns(2, gap="large")

col1.subheader("Predicted")
col2.subheader("Closest train SMILES")
for i, row in df_filtered.iterrows():
    caption_1 = f"True: {row['Class']}, Predicted: {row['Predicted value']}, P(1) = {row['P (class 1)']:0.3f}"
    caption_2 = f"p0_p1 discordance: {row['diff_p1_p0']:.3f}"
    col1.image(
        smi2img(row["Canon_SMILES"]),
#        use_column_width=True
    )
    col1.write(caption_1)
    col1.write(caption_2)

    smi = row["Closest Train SMILES"]
    tanimoto = row["Tanimoto Similarity"]
    # match the SMILES with the training set
    train_instances = train_df.query("Canon_SMILES == @smi")
    # get every protein in train_instances
    class_protein = train_instances[["Protein","Class"]].values.tolist()
    caption_a = f"Tanimoto similarity: {tanimoto:.2f}"
    caption_b = f"Train vals: {class_protein}"
    col2.image(
        smi2img(smi),
#        use_column_width=True
    )
    col2.write(caption_a)
    col2.write(caption_b)

    if train_instances.shape[0] > 2:
        # hacky way to improve alignment
        n_instances = train_instances.shape[0]
        n_new_lines = n_instances // 2
        for i in range(n_new_lines):
            col1.write("\n")

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
