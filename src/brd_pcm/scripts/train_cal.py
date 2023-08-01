# %% [markdown]
# This file is for training the model with calibration

# %%
import pickle
import bz2
import os
import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

# imblearn
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline

# custom functions
from brd_pcm.pcm_tools.ml import ros_by_protein_class, run_CVAP

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
# load test data and sort out the columns
X_test = pd.read_parquet(str(upstream[upstream_name]["X_test"]))
X_test = X_test.drop(columns=["Canon_SMILES", "Murcko_SMILES"])
y_test = pd.read_parquet(str(upstream[upstream_name]["y_test"]))
# make y_test a series too, although this probably matters less than for y_train
y_test = y_test.squeeze()

# %%
# load pickled imblearn pipeline (unfitted)
with open(upstream[upstream_name]["imblearn_pipe"], "rb") as f:
    pipe_clf = pickle.load(f)

# %%
# setup for cross Venn-ABERS predictors
cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_seed)
# get indices of cv splits (as a single variable)
cv_indices = [(train, test) for train, test in cv.split(X_train, y_train, train_groups)]

# %%
# check cv indices and pickle for later use
for i, (train, test) in enumerate(cv_indices):
    print(f"Fold {i}: {len(train)} train molecules, {len(test)} test molecules")
    print(train)

with open(product["cv_indices"], "wb") as f:
    pickle.dump(cv_indices, f)

# %%
# run CVAP, need to implement a way to store models
# note that the y_test index MUST be from 0 to len(y_test) - 1, otherwise the Venn-ABERS
# code doesn't work
if not os.path.exists(str(product["model"])):
    os.makedirs(str(product["model"]))

va_df = run_CVAP(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    clf=pipe_clf,
    cv_indices=cv_indices,
    groups=train_groups,
    rfe_cv=False,
    threshold=0.5,
    outdir=None,
    save_clf=True,
    clf_outdir=str(product["model"]),
)

# %%
va_df.head()

# %%
# save predictions
# make it more similar to the uncal_train.py output in terms of column names
pred_df = pd.read_parquet(
    str(upstream[upstream_name]["X_test"]),
    columns=["Canon_SMILES", "Protein", "Murcko_SMILES"],
)

pred_df = pd.concat([pred_df, va_df], axis=1)
pred_df = pred_df.rename(
    columns={"True value": "Class", "avg_single_prob": "P (class 1)"}
)

# %%
# save the dataframe
pred_df.to_csv(product["predictions"], index=False)
