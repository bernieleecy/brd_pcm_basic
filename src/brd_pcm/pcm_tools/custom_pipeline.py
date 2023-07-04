# Functions required to set up the training pipeline
# Moved here so I can use them in multiple notebooks
import pandas as pd

import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline


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
            verbose=True,
        )
    else:
        new_pipe = Pipeline(
            steps=[
                ("preprocessing", preprocess_ct),
                ("resample", sampler),
                ("drop", drop_col),
                ("classify", train_clf),
            ],
            verbose=True,
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
