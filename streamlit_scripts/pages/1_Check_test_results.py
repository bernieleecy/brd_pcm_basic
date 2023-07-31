import streamlit as st
from st_aggrid import AgGrid
import streamlit.components.v1 as components
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen
from rdkit.Chem import PandasTools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import recall_score, matthews_corrcoef
import mols2grid

st.title("Checking test set results")
st.markdown(
    """
    This page is for test set results only. If you made predictions on unseen
    data, please proceed to the next page (check served predictions).
    """
)

def smi2img(smi):
    mol = Chem.MolFromSmiles(smi)
    rdCoordGen.AddCoords(mol)
    img = Draw.MolToImage(mol, size=(300,300))
    return img


# load in data for test set, hard coded during testing
file = st.file_uploader(
    "Upload a CSV file containing test set results here", type="csv", key="test_set"
)

if file is not None:
    st.session_state["df_test"] = pd.read_csv(file)

if "df_test" in st.session_state:
    df_test = st.session_state["df_test"]

    # unique proteins, sorted alphabetically
    proteins = sorted(df_test["Protein"].unique())

    # check for misclassified instances
    df_test["Correct"] = df_test["Class"] == df_test["Predicted value"]

    # filter data by protein
    selected_protein = st.selectbox("Select a protein", proteins)

    # get some metrics
    df_test_filtered = df_test.query("Protein == @selected_protein")
    correct_ligands = df_test_filtered.query("Correct == True")
    incorrect_ligands = df_test_filtered.query("Correct == False")

    # sort by P (class 1)
    df_test_filtered = df_test_filtered.sort_values(by="P (class 1)", ascending=False)

    st.header(f"Test set information for {selected_protein}")

    # get a summary, starting with test set composition
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        label=f"Total ligands for {selected_protein}", value=len(df_test_filtered)
    )
    col_b.metric(label="Actives", value=len(df_test_filtered.query("Class == 1")))
    col_c.metric(label="Inactives", value=len(df_test_filtered.query("Class == 0")))

    # then go through some key metrics for each protein
    col_d, col_e, col_f = st.columns(3)
    sensitivity = recall_score(
        df_test_filtered["Class"], df_test_filtered["Predicted value"]
    )
    specificity = recall_score(
        df_test_filtered["Class"], df_test_filtered["Predicted value"], pos_label=0
    )
    mcc = matthews_corrcoef(
        df_test_filtered["Class"], df_test_filtered["Predicted value"]
    )
    col_d.metric(label="Sensitivity", value=f"{sensitivity:.3f}")
    col_e.metric(label="Specificity", value=f"{specificity:.3f}")
    try:
        col_f.metric(label="MCC", value=f"{mcc:.3f}")
    except:
        col_f.metric(label="MCC", value="N/A")

    # make confusion matrix if possible
    cm = confusion_matrix(
        df_test_filtered["Class"], df_test_filtered["Predicted value"]
    )

    # display confusion matrix
    st.subheader("Confusion matrix")
    try:
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Inactive", "Active"]
        )
        fig, ax = plt.subplots(figsize=(3,2.25), constrained_layout=True)
        disp.plot(ax=ax, cmap="rocket", colorbar=False)
        ax.set_xlabel("Predicted label", fontsize=8)
        ax.set_ylabel("True label", fontsize=8)
        ax.set_xticklabels(["Inactive", "Active"], fontsize=7)
        ax.set_yticklabels(["Inactive", "Active"], fontsize=7)
        st.pyplot(fig, use_container_width=False)
    except:
        st.write("Confusion matrix cannot be displayed.")

    st.header(f"Predictions for {selected_protein}")
    misclassified = st.checkbox("Show misclassified instances only")
    if misclassified:
        df_test_filtered = df_test_filtered.query("Correct == False")

    raw_html = mols2grid.display(
        df_test_filtered,
        smiles_col="Canon_SMILES",
        subset=["img", "Class", "Predicted value", "P (class 1)", "diff_p1_p0"],
        size=(225, 175),
        n_cols=3,
        tooltip=["Closest Train SMILES", "Tanimoto Similarity"],
        transform={
            "Class": lambda x: f"True class: {x}",
            "Predicted value": lambda x: f"Predicted class: {x}",
            "P (class 1)": lambda x: f"P (class 1): {x:.3f}",
            "diff_p1_p0": lambda x: f"VA interval: {x:.3f}",
            "Tanimoto Similarity": lambda x: f"{x:.3f}",
        },
    )._repr_html_()
    components.html(raw_html, height=800, scrolling=True)

    st.header("Compare against training instances")
    # upload training data
    X_train = st.file_uploader(
        "Upload X_train parquet here", type="parquet", key="X_train"
    )
    y_train = st.file_uploader(
        "Upload y_train parquet here", type="parquet", key="y_train"
    )

    if X_train is not None and y_train is not None:
        X_train = pd.read_parquet(X_train)
        y_train = pd.read_parquet(y_train)
        training_df = pd.concat([X_train, y_train], axis=1)
        st.session_state["training_df"] = training_df

    if "training_df" in st.session_state:
        # only do this for the filtered df here!
        # for each row, find the closest training instance
        # using iterrows is not ideal, but it works for now
        if len(df_test_filtered) > 10:
            st.warning(
                "There are more than 10 ligands to display. Please use the slider to go through the pages"
            )
            page = st.slider("Page", 1, len(df_test_filtered) // 10 + 1, 1)
            df_test_filtered = df_test_filtered.iloc[(page - 1) * 10 : page * 10]

        for i, (index, row) in enumerate(df_test_filtered.iterrows()):
            # initialize columns each time to get better alignment
            col1, col2, col3 = st.columns(3)
            if i == 0:
                col1.subheader("Test")
                col2.subheader("Closest train")
                col3.subheader("Info")

            # first column gets the test instance
            col1.image(smi2img(row["Canon_SMILES"]))

            # second column gets the closest training instance
            smi = row["Closest Train SMILES"]
            col2.image(smi2img(row["Closest Train SMILES"]))

            # third column is for information
            train_instances = st.session_state["training_df"].query(
                "Canon_SMILES == @smi"
            )
            class_protein = train_instances[["Protein", "Class"]].values.tolist()
            # if misclassfied, print
            if not row["Correct"]:
                col3.markdown(f":red[**MISCLASSIFIED!**]")
            col3.write(f"True val: {row['Class']}")
            col3.write(f"Predicted val: {row['Predicted value']} ({row['P (class 1)']:.3f})")
            col3.write(f"VA interval: {row['diff_p1_p0']:.3f}")
            col3.write(f"Closest train Tanimoto: {row['Tanimoto Similarity']:.3f}")
            col3.write(f"{class_protein}")
            st.write("---")
