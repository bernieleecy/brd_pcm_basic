# %% [markdown]
# This file is for serving predictions with calibration

# %%
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

# fix random seed for reproducibility
random_seed = 13579
rng = np.random.RandomState(random_seed)
log.info(f"Random seed: {random_seed}")

# %% tags=["parameters"]
upstream = None
product = None
X_train_data = None
y_train_data = None
cv_data = None
model_folder = None

# %%
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
# load the data for predictions and sort out the columns
all_data = pd.read_parquet(str(upstream[upstream_name]["data"]))
pcm_data = all_data.drop(columns=["Canon_SMILES", "Class"])

# %%
# load the X_train and y_train data (required to set up calibration)
X_train = pd.read_parquet(X_train_data)
X_train = X_train.drop(columns=["Canon_SMILES", "Murcko_SMILES"])
y_train = pd.read_parquet(y_train_data)
y_train = y_train.squeeze()

# %%
# load CV indices
with open(cv_data, "rb") as f:
    cv_indices = pickle.load(f)

# %%
# make predictions
va_df = predict_CVAP(
    X_train=X_train,
    y_train=y_train,
    X_test=pcm_data,
    clf_dir=model_folder,
    cv_indices=cv_indices,
)

# %%
va_df.head()

# %%
# save predictions
# make it more similar to the uncal_train.py output in terms of column names
pred_df = all_data.loc[:, ["Canon_SMILES", "Protein"]]
pred_df = pd.concat([pred_df, va_df], axis=1)
pred_df = pred_df.rename(
    columns={"True value": "Class", "avg_single_prob": "P (class 1)"}
)

pred_df.to_csv(product["predictions"], index=False)
