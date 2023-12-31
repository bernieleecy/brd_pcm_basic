# %% [markdown]
# This file is for training the model without calibration

# %%
import pickle
import bz2
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
from brd_pcm.utils.ml import ros_by_protein_class

# logging
import logging
logging.basicConfig(format="%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)

# %% tags=["parameters"]
upstream = None
product = None
random_seed = None

# %%
# fix random seed for reproducibility
random_seed = int(random_seed)
rng = np.random.RandomState(random_seed)
log.info(f"Random seed: {random_seed}")

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
# Fit model on all training data
pipe_clf.fit(X_train, y_train)
# pickle the pipeline (where the RF model is still uncalibrated)
with bz2.open(product["model"], "wb") as f:
    pickle.dump(pipe_clf, f)

# get feature names
feature_names = pipe_clf.named_steps["preprocessing"].get_feature_names_out()
# get discrete features
discrete_feats = [feat.split("__")[-1] for feat in feature_names if "discrete" in feat]
continuous_feats = [feat.split("__")[-1] for feat in feature_names if "continuous" in feat]
log.info(discrete_feats[:5], len(discrete_feats))
log.info(continuous_feats[:5], len(continuous_feats))

with open(product["lig_feat_names"], "wb") as f:
    pickle.dump(discrete_feats, f)
with open(product["prot_feat_names"], "wb") as f:
    pickle.dump(continuous_feats, f)

# %%
# Predict on test data
y_pred = pipe_clf.predict(X_test)
y_pred_proba = pipe_clf.predict_proba(X_test)

# %%
# save predictions
pred_df = pd.read_parquet(
    str(upstream[upstream_name]["X_test"]),
    columns=["Canon_SMILES", "Protein", "Murcko_SMILES"],
)
pred_df["Class"] = y_test
pred_df["Predicted value"] = y_pred
pred_df["P (class 0)"] = y_pred_proba[:, 0]
pred_df["P (class 1)"] = y_pred_proba[:, 1]

pred_df.to_csv(product["predictions"], index=False)
