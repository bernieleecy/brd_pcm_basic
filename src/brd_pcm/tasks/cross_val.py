import pickle
import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score, matthews_corrcoef

# imblearn
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline

# custom functions
from brd_pcm.utils.ml import ros_by_protein_class


def cross_val(upstream, product, type, random_seed):
    # fix random seed for reproducibility
    random_seed = int(random_seed)
    rng = np.random.RandomState(random_seed)

    # Get the upstream name (assumes single upstream here)
    upstream_name = list(upstream)[0]

    # load training data and sort out the columns, test set not needed here
    X_train = pd.read_parquet(str(upstream[upstream_name]["X_train"]))
    train_groups = X_train["Murcko_SMILES"].copy()
    X_train = X_train.drop(columns=["Canon_SMILES", "Murcko_SMILES"])
    y_train = pd.read_parquet(str(upstream[upstream_name]["y_train"]))
    # y_train must be a series
    y_train = y_train.squeeze()

    # load pickled imblearn pipeline (unfitted)
    with open(upstream[upstream_name]["imblearn_pipe"], "rb") as f:
        pipe_clf = pickle.load(f)

    scoring = {
        "roc_auc": "roc_auc",
        "sensitivity": make_scorer(recall_score, pos_label=1),
        "specificity": make_scorer(recall_score, pos_label=0),
        "mcc": make_scorer(matthews_corrcoef),
    }

    # Check cross-validation scores
    if type == "stratified":
        cross_val_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
        cross_val_scores = cross_validate(
            pipe_clf,
            X=X_train,
            y=y_train,
            groups=train_groups,
            cv=cross_val_cv,
            scoring=scoring,
        )
    elif type == "random":
        # must shuffle if the input came from a StratifiedGroupKFold split
        cross_val_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        cross_val_scores = cross_validate(
            pipe_clf,
            X=X_train,
            y=y_train,
            cv=cross_val_cv,
            scoring=scoring,
        )
    else:
        raise ValueError("type must be either 'stratified' or 'random'")

    # save cross_val_scores
    cross_val_scores_df = pd.DataFrame(cross_val_scores)
    mean_scores = cross_val_scores_df.mean()
    std_scores = cross_val_scores_df.std()
    cross_val_scores_df.loc["mean"] = mean_scores
    cross_val_scores_df.loc["std"] = std_scores
    # transpose df before saving, keep the index
    cross_val_scores_df.T.to_csv(product["cross_val"])
