# %% [markdown]
# This file is for training the model without calibration

# %%
import pickle
import numpy as np
import pandas as pd

# sklearn
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer, recall_score

# imblearn
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline

# custom functions
from pythia.pcm_tools import SplitData
from brd_pcm.pcm_tools.custom_pipeline import ros_by_protein_class

# logging
import logging

logging.basicConfig(format="%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)

# fix random seed for reproducibility
random_seed = 13579
rng = np.random.RandomState(random_seed)
log.info(f"Random seed: {random_seed}")

# %% tags=["parameters"]
upstream = None
product = None

# %%
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
# load training data and sort out the columns, test set not needed here
X_train = pd.read_parquet(str(upstream[upstream_name]["X_train"]))
train_groups = X_train["Murcko_SMILES"].copy()
X_train = X_train.drop(columns=["Canon_SMILES", "Murcko_SMILES"])
y_train = pd.read_parquet(str(upstream[upstream_name]["y_train"]))
# y_train must be a series
y_train = y_train.squeeze()

# %%
# load pickled imblearn pipeline (unfitted)
with open(upstream[upstream_name]["imblearn_pipe"], "rb") as f:
    pipe_clf = pickle.load(f)

# %%
# Check cross-validation scores
cross_val_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
scoring = {
    "roc_auc": "roc_auc",
    "sensitivity": make_scorer(recall_score, pos_label=1),
    "specificity": make_scorer(recall_score, pos_label=0),
}

cross_val_scores = cross_validate(
    pipe_clf,
    X=X_train,
    y=y_train,
    groups=train_groups,
    cv=cross_val_cv,
    scoring=scoring,
)

# %%
# save cross_val_scores
cross_val_scores_df = pd.DataFrame(cross_val_scores)
mean_scores = cross_val_scores_df.mean()
std_scores = cross_val_scores_df.std()
cross_val_scores_df.loc["mean"] = mean_scores
cross_val_scores_df.loc["std"] = std_scores
# transpose df before saving, keep the index
cross_val_scores_df.T.to_csv(product["cross_val"])

# %%
# main output
cross_val_scores_df.T
