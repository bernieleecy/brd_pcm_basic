# Functions required to set up the training pipeline
# Moved here so I can use them in multiple notebooks
import os
import pickle
import bz2
import numpy as np
import pandas as pd
from scipy.stats import gmean

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

# for venn-abers calibration
from .VennABERS import ScoresToMultiProbs

import logging

log = logging.getLogger(__name__)


def make_imblearn_pipe(
    train_lig_feats,
    train_prot_feats,
    sampler,
    train_clf,
    var_thres=0.95,
    scaler=MinMaxScaler(),
    cachedir=None,
):
    """Function to generate pipelines
    Supports ligand fingerprints (discrete feats) and continuous protein
    features.

    Args:
        train_lig_feats (list): List of ligand features
        train_prot_feats (list): List of protein features
        sampler (imblearn sampler): Sampler to use for resampling. Usually a function sampler.
        train_clf (sklearn estimator): Classifier to use for training.
        var_thres (float, optional): Variance threshold for removing low
            variance ligand fingerprints. Defaults to 0.95.
        scaler (sklearn scaler, optional): Scaler to use for protein features. Defaults
            to MinMaxScaler().
        cachedir (str, optional): Path to cache directory. Defaults to None.
    """
    var_thres_lig = VarianceThreshold(threshold=(var_thres * (1 - var_thres)))
    var_thres_prot = VarianceThreshold()

    # step 1: remove low variance ligand features and zero variance protein features
    # also scale protein features
    discrete_feats_cut = Pipeline(steps=[("remove_low_var", var_thres_lig)])
    cont_feats_cut = Pipeline(
        steps=[
            ("remove_zero_var", var_thres_prot),
            ("scale_feat", scaler),
        ]
    )
    preprocess_ct = ColumnTransformer(
        transformers=[
            ("discrete", discrete_feats_cut, train_lig_feats),
            ("continuous", cont_feats_cut, train_prot_feats),
        ],
        remainder="passthrough",
    )

    # this is mandatory to drop the protein column
    drop_col = ColumnTransformer(
        transformers=[("drop_col", "drop", "remainder__Protein")],
        remainder="passthrough",
    )

    # assemble pipeline, step 2 is for resampling, step 4 is the classifier to use for training
    if cachedir is not None:
        new_pipe = Pipeline(
            steps=[
                ("preprocessing", preprocess_ct),
                ("resample", sampler),
                ("drop", drop_col),
                ("classify", train_clf),
            ],
            memory=cachedir,
        )
    else:
        new_pipe = Pipeline(
            steps=[
                ("preprocessing", preprocess_ct),
                ("resample", sampler),
                ("drop", drop_col),
                ("classify", train_clf),
            ],
        )
    return new_pipe


def ros_by_protein_class(X_train, y_train, random_seed):
    """Oversample by protein class using RandomOverSampler from imblearn.
    After preprocessing, the protein column ("remainder") will be the final column
    """
    unique_proteins = X_train.iloc[:, -1].unique()

    x_dfs = []
    y_dfs = []

    for protein in unique_proteins:
        single_prot = X_train[X_train.iloc[:, -1] == protein]
        y_train_prot = y_train.loc[single_prot.index]

        # check for a single class having no samples
        if len(y_train_prot.unique()) == 1:
            x_dfs.append(single_prot)
            y_dfs.append(y_train_prot)
        else:
            ros = RandomOverSampler(random_state=random_seed)
            oversample_X, oversample_y = ros.fit_resample(single_prot, y_train_prot)
            x_dfs.append(oversample_X)
            y_dfs.append(oversample_y)

    # make a new X_train and y_train df and reset index
    X_train = pd.concat(x_dfs)
    y_train = pd.concat(y_dfs)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    return X_train, y_train


# Venn-ABERS predictors
def run_CVAP(
    X_train,
    y_train,
    X_test,
    y_test,
    clf,
    cv_indices,
    groups=None,
    rfe_cv=False,
    rfe_cv_name="RFECV",
    threshold=0.5,
    outdir=None,
    save_clf=False,
    clf_outdir=None,
):
    """Function to run cross venn-abers predictions on test set
    Based partially on
    https://github.com/valeman/Multi-class-probabilistic-classification (although this
    function is for binary classification only)

    Args:
        X_train (pd DataFrame): DataFrame containing training data (with features).
        y_train (pd Series): Series containing training labels.
        X_test (pd DataFrame): DataFrame containing test data (with features).
        y_test (pd Series): Series containing test labels.
        clf (sklearn estimator): Classifier to use for training.
        cv_indices (list): List of tuples containing train/test indices for each CV split.
        groups (pd Series, optional): Series containing group labels. Defaults to None.
        rfe_cv (bool, optional): Whether a pipeline with RFECV is being using as clf. Defaults to False.
        rfe_cv_name (str, optional): Name of RFECV step in pipeline. Defaults to "RFECV".
        threshold (float, optional): Threshold for classification. Defaults to 0.5.
        outdir (str, optional): Path to output directory. Defaults to None.
        save_clf (bool, optional): Whether to save the classifiers. Defaults to False.
        clf_outdir (str, optional): Path to output directory for saving
            classifiers. Defaults to None.

    Returns:
        DataFrame: DataFrame containing true val, predicted val, and venn-abers
            predictions on the test set
    """
    test_length = len(X_test)
    per_fold_p0 = np.zeros((test_length, len(cv_indices)))
    per_fold_p1 = np.zeros((test_length, len(cv_indices)))
    per_fold_single = np.zeros((test_length, len(cv_indices)))

    for i, (proper_train_idx, cal_idx) in enumerate(cv_indices):
        X_proper_train = X_train.iloc[proper_train_idx]
        y_proper_train = y_train.iloc[proper_train_idx]
        X_cal = X_train.iloc[cal_idx]
        y_cal = y_train.iloc[cal_idx]

        # check if groups are passed, if so, pass them to the RFECV fit
        if rfe_cv and groups is not None:
            fit_params = {f"{rfe_cv_name}__groups": groups.iloc[proper_train_idx]}
            clf.fit(X_proper_train, y_proper_train, **fit_params)
        else:
            clf.fit(X_proper_train, y_proper_train)

        if save_clf:
            with bz2.open(f"{clf_outdir}/clf_{i+1}.pkl.bz2", "wb") as f:
                pickle.dump(clf, f)

        # get probabilities for calibration set and test set
        cal_pred_proba_c1 = clf.predict_proba(X_cal)[:, 1]
        test_pred_proba_c1 = clf.predict_proba(X_test)[:, 1]

        # zip cal scores and true cal labels
        cal_scores_labels = list(zip(cal_pred_proba_c1, y_cal))

        # do venn-abers calibration
        p0, p1 = ScoresToMultiProbs(cal_scores_labels, test_pred_proba_c1)
        per_fold_p0[:, i] = p0
        per_fold_p1[:, i] = p1

        single_prob = p1 / (1 - p0 + p1)
        per_fold_single[:, i] = single_prob

    # write per-fold data if desired
    if outdir is not None:
        fold_names = [f"fold_{i+1}" for i in range(len(cv_indices))]
        p0_fold_data = pd.DataFrame(per_fold_p0, index=None, columns=fold_names)
        p1_fold_data = pd.DataFrame(per_fold_p1, index=None, columns=fold_names)
        single_fold_data = pd.DataFrame(per_fold_single, index=None, columns=fold_names)
        p0_fold_data.to_csv(os.path.join(outdir, "p0_fold_data.csv"), index=False)
        p1_fold_data.to_csv(os.path.join(outdir, "p1_fold_data.csv"), index=False)
        single_fold_data.to_csv(
            os.path.join(outdir, "single_fold_data.csv"), index=False
        )

    # use gmean to combine predictions, as proposed by Vovk 2015
    gmean_1p0 = gmean((1 - per_fold_p0), axis=1)
    gmean_p1 = gmean(per_fold_p1, axis=1)
    avg_single = gmean_p1 / (gmean_1p0 + gmean_p1)
    y_pred = np.where(avg_single > threshold, 1, 0)

    # save the arithmetic means of p0 and p1
    amean_p0 = np.mean(per_fold_p0, axis=1)
    amean_p1 = np.mean(per_fold_p1, axis=1)
    diff_p1_p0 = amean_p1 - amean_p0

    test_preds = pd.DataFrame(
        [y_test, y_pred, avg_single, amean_p0, amean_p1, diff_p1_p0]
    ).T
    test_preds.columns = [
        "True value",
        "Predicted value",
        "avg_single_prob",
        "amean_p0",
        "amean_p1",
        "diff_p1_p0",
    ]
    # set true and predicted values to int
    test_preds["True value"] = test_preds["True value"].astype(int)
    test_preds["Predicted value"] = test_preds["Predicted value"].astype(int)
    return test_preds


def predict_CVAP(
    X_train,
    y_train,
    X_test,
    clf_dir,
    cv_indices,
    threshold=0.5,
    outdir=None,
):
    """Function to run cross venn-abers predictions (assuming classifiers are trained)

    Args:
        X_train (pd DataFrame): DataFrame containing training data (with features).
        y_train (pd Series): Series containing training labels.
        X_test (pd DataFrame): DataFrame containing test data (with features).
        clf_dir (str): Path to directory containing pickled classifiers.
        cv_indices (list): List of tuples containing train/test indices for each CV split.
        threshold (float, optional): Threshold for classification. Defaults to 0.5.
        outdir (str, optional): Path to output directory. Defaults to None.

    Returns:
        DataFrame: DataFrame containing predicted val, and the calibrated probabilities.
    """
    test_length = len(X_test)
    per_fold_p0 = np.zeros((test_length, len(cv_indices)))
    per_fold_p1 = np.zeros((test_length, len(cv_indices)))
    per_fold_single = np.zeros((test_length, len(cv_indices)))

    for i, (proper_train_idx, cal_idx) in enumerate(cv_indices):
        X_cal = X_train.iloc[cal_idx]
        y_cal = y_train.iloc[cal_idx]

        # load trained classifier (trained on this set of proper_train_idx)
        with bz2.open(f"{clf_dir}/clf_{i+1}.pkl.bz2", "rb") as f:
            clf = pickle.load(f)

        # get probabilities for calibration set and test set
        cal_pred_proba_c1 = clf.predict_proba(X_cal)[:, 1]
        test_pred_proba_c1 = clf.predict_proba(X_test)[:, 1]

        # zip cal scores and true cal labels
        cal_scores_labels = list(zip(cal_pred_proba_c1, y_cal))

        # do venn-abers calibration
        p0, p1 = ScoresToMultiProbs(cal_scores_labels, test_pred_proba_c1)
        per_fold_p0[:, i] = p0
        per_fold_p1[:, i] = p1

        single_prob = p1 / (1 - p0 + p1)
        per_fold_single[:, i] = single_prob

    # write per-fold data if desired
    if outdir is not None:
        fold_names = [f"fold_{i+1}" for i in range(len(cv_indices))]
        p0_fold_data = pd.DataFrame(per_fold_p0, index=None, columns=fold_names)
        p1_fold_data = pd.DataFrame(per_fold_p1, index=None, columns=fold_names)
        single_fold_data = pd.DataFrame(per_fold_single, index=None, columns=fold_names)
        p0_fold_data.to_csv(os.path.join(outdir, "p0_fold_data.csv"), index=False)
        p1_fold_data.to_csv(os.path.join(outdir, "p1_fold_data.csv"), index=False)
        single_fold_data.to_csv(
            os.path.join(outdir, "single_fold_data.csv"), index=False
        )

    # use gmean to combine predictions, as proposed by Vovk 2015
    gmean_1p0 = gmean((1 - per_fold_p0), axis=1)
    gmean_p1 = gmean(per_fold_p1, axis=1)
    avg_single = gmean_p1 / (gmean_1p0 + gmean_p1)
    y_pred = np.where(avg_single > threshold, 1, 0)

    # save the arithmetic means of p0 and p1
    amean_p0 = np.mean(per_fold_p0, axis=1)
    amean_p1 = np.mean(per_fold_p1, axis=1)
    diff_p1_p0 = amean_p1 - amean_p0

    test_preds = pd.DataFrame([y_pred, avg_single, amean_p0, amean_p1, diff_p1_p0]).T
    test_preds.columns = [
        "Predicted value",
        "avg_single_prob",
        "amean_p0",
        "amean_p1",
        "diff_p1_p0",
    ]
    # set predicted values to int
    test_preds["Predicted value"] = test_preds["Predicted value"].astype(int)
    return test_preds
