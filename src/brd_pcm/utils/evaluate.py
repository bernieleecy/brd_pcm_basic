# tools for PCM modelling
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    auc,
)
from scipy.spatial.distance import cdist
from imblearn.metrics import geometric_mean_score

import logging

log = logging.getLogger(__name__)


def find_similar_train_ligand(
    train_df,
    test_df,
    smiles_col="Canon_SMILES",
    radius=3,
    nbits=1024,
    chirality=True,
    remove_dup=False,
):
    """Find the most similar train ligand to each test ligand.
    By default, this checks every prediction against unique train ligands, but
    I can turn on remove_dup to only output unique test SMILES strings.
    Because it uses unique train ligands, I can't attach class information.

    Args:
        train_df (pd DataFrame): DataFrame with shape (n_samples, n_cols). Must
            contain a column with ligand SMILES and Class.
        test_df (pd DataFrame): DataFrame with shape (n_samples, n_cols). Must
            contain a column with ligand SMILES.
        smiles_col (str): Name of column containing SMILES strings. Defaults to
            "Canon_SMILES", assumed to be the same in both dfs.
        radius (int): Radius of Morgan fingerprint. Defaults to 3.
        nbits (int): Number of bits in Morgan fingerprint. Defaults to 1024.
        chirality (bool): Whether to use chirality in Morgan fingerprint.
            Defaults to True.
        remove_dup (bool): Whether to remove duplicate test points. Defaults to
            False.

    Returns:
        most_similar_df (pd DataFrame): DataFrame with shape (n_samples, 3).
    """
    unique_train_mols = [
        Chem.MolFromSmiles(s, sanitize=True) for s in train_df[smiles_col].unique()
    ]
    # check every test point, even if it is a repeat
    test_mols = [Chem.MolFromSmiles(s, sanitize=True) for s in test_df[smiles_col]]

    train_fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            m, radius=radius, nBits=nbits, useChirality=chirality
        )
        for m in unique_train_mols
    ]
    test_fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            m, radius=radius, nBits=nbits, useChirality=chirality
        )
        for m in test_mols
    ]

    most_similar_data = []

    for i, fp in enumerate(test_fps):
        scores = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        # get the index of the max score
        max_score_idx = np.argmax(scores)
        # get the max score
        max_score = scores[max_score_idx]
        # write test smi, train smi, max score, max score train point to df
        most_similar_data.append(
            [
                test_df.iloc[i][smiles_col],
                train_df[smiles_col].unique()[max_score_idx],
                max_score,
            ]
        )

    most_similar_df = pd.DataFrame(
        most_similar_data,
        columns=["Test SMILES", "Closest Train SMILES", "Tanimoto Similarity"],
    )

    if remove_dup:
        most_similar_df = most_similar_df.drop_duplicates(subset=["Test SMILES"])

    return most_similar_df


def find_similar_train_ligand_fps(
    train_df,
    test_df,
    smiles_col="Canon_SMILES",
    radius=3,
    nbits=1024,
    chirality=True,
):
    """Find the most similar train ligand to each test ligand.
    This version is for use when only the test fingerprints are available (i.e. no
    SMILES data).
    Because it uses unique train ligands, I can't attach class information.

    Args:
        train_df (pd DataFrame): DataFrame with shape (n_samples, n_cols). Must
            contain a column with ligand SMILES and Class.
        test_df (pd DataFrame): DataFrame with shape (n_samples, n_cols). Contains
            ligand fingerprints.
        smiles_col (str): Name of column containing SMILES strings. Defaults to
            "Canon_SMILES", assumed to be the same in both dfs.
        radius (int): Radius of Morgan fingerprint. Defaults to 3.
        nbits (int): Number of bits in Morgan fingerprint. Defaults to 1024.
        chirality (bool): Whether to use chirality in Morgan fingerprint.
            Defaults to True.
        remove_dup (bool): Whether to remove duplicate test points. Defaults to
            False.

    Returns:
        most_similar_df (pd DataFrame): DataFrame with shape (n_samples, 3).
    """
    unique_train_mols = [
        Chem.MolFromSmiles(s, sanitize=True) for s in train_df[smiles_col].unique()
    ]
    train_fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            m, radius=radius, nBits=nbits, useChirality=chirality
        )
        for m in unique_train_mols
    ]

    fp_cols = [str(i) for i in range(nbits)]
    test_fps = test_df.loc[:, fp_cols].to_numpy()

    most_similar_data = []

    for i, fp in enumerate(test_fps):
        # convert test fps to explictbitvect
        # based on https://iwatobipen.wordpress.com/2019/02/08/convert-fingerprint-to-numpy-array-and-conver-numpy-array-to-fingerprint-rdkit-memorandum/
        fp_str = "".join(fp.astype(str))
        test_fp = DataStructs.cDataStructs.CreateFromBitString(fp_str)
        scores = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        # get the index of the max score
        max_score_idx = np.argmax(scores)
        # get the max score
        max_score = scores[max_score_idx]
        # get closest train smiles and max score
        most_similar_data.append(
            [
                train_df[smiles_col].unique()[max_score_idx],
                max_score,
            ]
        )

    # no test SMILES here
    most_similar_df = pd.DataFrame(
        most_similar_data,
        columns=["Closest Train SMILES", "Tanimoto Similarity"],
    )

    return most_similar_df


def get_pr_auc(y_true, y_pred, pos_label=1):
    """Get the precision-recall AUC.

    Args:
        y_true (np array): Array of true labels.
        y_pred (np array): Array of predicted labels.
        pos_label (int, optional): Positive label. Defaults to 1.

    Returns:
        float: Precision-recall AUC
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=pos_label)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_key_cmetrics(y_true, y_pred, y_pred_proba=None, pos_label=1):
    """Get key metrics for binary classification

    Args:
        y_true (np array): Array of true labels.
        y_pred (np array): Array of predicted labels.
        y_pred_proba (np array, optional): Array of predicted probabilities. Defaults to
            None.
        pos_label (int, optional): Positive label. Defaults to 1.

    Returns:
        dict: Dictionary containing the metrics.
    """
    metrics = {}

    # get MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    metrics["mcc"] = mcc

    # get precision
    precision = precision_score(y_true, y_pred, pos_label=pos_label)
    metrics["precision"] = precision

    # get sensitivity (recall) and specificity (tnr)
    recall = recall_score(y_true, y_pred)
    tnr = recall_score(y_true, y_pred, pos_label=0)
    metrics["recall"] = recall
    metrics["tnr"] = tnr

    # get geometric mean (of sensitivity and specificity)
    gmean = geometric_mean_score(y_true, y_pred)
    metrics["gmean"] = gmean

    # get F1
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    metrics["f1"] = f1

    # get balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    metrics["balanced_acc"] = balanced_acc

    # these metrics require predicted probabilities
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except:
            roc_auc = np.nan
        try:
            pr_auc = get_pr_auc(y_true, y_pred_proba, pos_label=pos_label)
        except:
            pr_auc = np.nan
        metrics["roc_auc"] = roc_auc
        metrics["pr_auc"] = pr_auc

    return metrics


# Parse classification results by protein
def get_by_protein_preds(
    test_info_df,
    y_pred=None,
    y_pred_proba=None,
    protein_col="Protein",
    class_col="Class",
    outfile=None,
    va_pred=False,
):
    """Get the predictions by protein.

    Args:
        test_info_df (pd DataFrame): DataFrame containing information about the test set.
            Needs to contain protein data. Class data is assumed to be the last column.
        y_pred (np array): Array of predictions.
        y_pred_proba (np array): Array of prediction probabilities (class 0 and
            class 1 in binary classification).
        protein_col (str): Name of the column containing protein data.
        class_col (str): Name of the column containing class data.
        outfile (str, optional): Path to save the results to. Defaults to None.
        va_pred (bool, optional): Whether the predictions have Venn-ABERS calibration applied.

    Returns:
        dict: Dictionary containing the results (protein with roc_auc, recall and tnr)
    """
    # don't modify the original dataframe
    test_info = test_info_df.copy()

    # check if y_pred and y_pred_proba are not None (i.e. test info already in the right format)
    if y_pred is None and y_pred_proba is None:
        pass
    elif y_pred is None or y_pred_proba is None:
        raise ValueError("y_pred and y_pred_proba need to be both None or not None")
    else:
        test_info["Predicted value"] = y_pred
        if va_pred:
            test_info["P (class 1)"] = y_pred_proba
        else:
            test_info["P (class 0)"] = y_pred_proba[:, 0]
            test_info["P (class 1)"] = y_pred_proba[:, 1]

        if outfile is not None:
            test_info.to_csv(outfile, index=False)

    # get the unique proteins
    unique_proteins = test_info[protein_col].unique()

    # create a dictionary to store the results
    protein_dict = {}

    # loop through the proteins
    for protein in unique_proteins:
        df_protein = test_info.query(f"{protein_col} == '{protein}'")
        try:
            roc_auc = roc_auc_score(df_protein[class_col], df_protein["P (class 1)"])
        except:
            roc_auc = np.nan
        # compute recall (senstivity) and tnr (specificity)
        recall = recall_score(df_protein[class_col], df_protein["Predicted value"])
        tnr = recall_score(
            df_protein[class_col], df_protein["Predicted value"], pos_label=0
        )

        metrics = [roc_auc, recall, tnr]

        protein_dict[protein] = metrics

    by_protein_df = pd.DataFrame.from_dict(protein_dict, orient="index")
    by_protein_df.columns = ["ROC AUC", "Sensitivity", "Specificity"]
    # sort index
    by_protein_df = by_protein_df.sort_index()

    if outfile is not None:
        by_protein_df.to_csv(outfile.replace(".csv", "_by_protein_summary.csv"))

    return by_protein_df


def plot_by_protein_preds(by_protein_df, ax=None):
    """Plot the by-protein classification results (roc_auc, sensitivity (recall),
    specificity

    Args:
        by_protein_df (pd DataFrame): DataFrame containing the by-protein results
        ax (matplotlib axis): Axis to plot on. Defaults to None.
    """
    if ax is None:
        plt.gca()

    # sort by index
    by_protein_df = by_protein_df.sort_index()
    by_protein_df.plot(kind="bar", width=0.7, ax=ax)
    ax.set(ylabel="Value")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncols=3)
    ax.tick_params(axis="x", labelsize=11)

    return ax
