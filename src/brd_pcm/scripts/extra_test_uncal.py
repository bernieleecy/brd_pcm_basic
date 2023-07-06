# %% [markdown]
# This file is for serving predictions without calibration

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
from brd_pcm.pcm_tools.ml import ros_by_protein_class

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
path_to_model = None

# %%
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
# load the data for predictions and sort out the columns
all_data = pd.read_parquet(str(upstream[upstream_name]["data"]))
pcm_data = all_data.drop(columns=["Canon_SMILES", "Class"])

# %%
# load the model
with bz2.open(path_to_model, "rb") as f:
    model = pickle.load(f)

# %%
# make predictions
y_pred = model.predict(pcm_data)
y_pred_proba = model.predict_proba(pcm_data)

# %% save predictions
pred_df = all_data.loc[:,["Canon_SMILES", "Protein", "Class"]]
pred_df["Predicted value"] = y_pred
pred_df["P (class 0)"] = y_pred_proba[:, 0]
pred_df["P (class 1)"] = y_pred_proba[:, 1]

pred_df.to_csv(product["predictions"], index=False)
