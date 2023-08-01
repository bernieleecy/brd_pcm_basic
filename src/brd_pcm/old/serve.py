import pickle
import bz2
import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# imblearn
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline

# custom functions
from brd_pcm.pcm_tools.ml import ros_by_protein_class, predict_CVAP

# logging
import logging

logging.basicConfig(format="%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)


def serve_uncal(upstream, product, path_to_model, from_fps=False):
    """For serving uncalibrated predictions"""

    upstream_name = list(upstream)[0]
    # load the data for predictions and sort out the columns
    all_data = pd.read_parquet(str(upstream[upstream_name]["data"]))

    if not from_fps:
        pcm_data = all_data.drop(columns=["Canon_SMILES"])
    else:
        # make compatible w/ UCLA data
        pcm_data = all_data.drop(columns=["Running number"])

    # load the model
    with bz2.open(path_to_model, "rb") as f:
        model = pickle.load(f)

    # make predictions
    y_pred = model.predict(pcm_data)
    y_pred_proba = model.predict_proba(pcm_data)

    # save predictions
    if from_fps:
        # loc-ing in this way gives a dataframe rather than a series
        pred_df = all_data.loc[:, ["Running number", "Protein"]]
    else:
        pred_df = all_data.loc[:, ["Canon_SMILES", "Protein"]]
    pred_df["Predicted value"] = y_pred
    pred_df["P (class 0)"] = y_pred_proba[:, 0]
    pred_df["P (class 1)"] = y_pred_proba[:, 1]

    if str(product["predictions"]).endswith(".parquet"):
        pred_df.to_parquet(product["predictions"])
    else:
        pred_df.to_csv(product["predictions"], index=False)


def serve_cal(
    upstream, product, X_train_data, y_train_data, cv_data, model_folder, from_fps=False
):
    """
    For serving calibrated predictions
    """
    # Get the upstream name (assumes single upstream here)
    upstream_name = list(upstream)[0]
    # load the data for predictions and sort out the columns
    all_data = pd.read_parquet(str(upstream[upstream_name]["data"]))

    # load the X_train and y_train data (required to set up calibration)
    X_train = pd.read_parquet(X_train_data)
    X_train = X_train.drop(columns=["Canon_SMILES", "Murcko_SMILES"])
    y_train = pd.read_parquet(y_train_data)
    y_train = y_train.squeeze()

    # remove unneeded columns
    if not from_fps:
        pcm_data = all_data.drop(columns=["Canon_SMILES"])
    else:
        pcm_data = all_data.drop(columns=["Running number"])

    # load the cross-validation indices
    with open(cv_data, "rb") as f:
        cv_indices = pickle.load(f)

    # make predictions
    va_df = predict_CVAP(
        X_train=X_train,
        y_train=y_train,
        X_test=pcm_data,
        clf_dir=model_folder,
        cv_indices=cv_indices,
    )

    # save predictions
    # make it more similar to the uncal_train.py output in terms of column names
    if from_fps:
        # loc-ing in this way gives a dataframe rather than a series
        pred_df = all_data.loc[:, ["Running number", "Protein"]]
    else:
        pred_df = all_data.loc[:, ["Canon_SMILES", "Protein"]]
    pred_df = pd.concat([pred_df, va_df], axis=1)
    pred_df = pred_df.rename(columns={"avg_single_prob": "P (class 1)"})

    if str(product["predictions"]).endswith(".parquet"):
        pred_df.to_parquet(product["predictions"])
    else:
        pred_df.to_csv(product["predictions"], index=False)
