# %% [markdown]
# This file is for flagging misclassified examples in test sets (calibrated preds only)

# %%
import numpy as np
import pandas as pd

from brd_pcm.utils.evaluate import find_similar_train_ligand

# logging
import logging

logging.basicConfig(format="%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)

# set display and plotting preferences
pd.options.display.float_format = "{:.3f}".format

# %% tags=["parameters"]
upstream = None
product = None
X_train_data = None

# %%
# load data
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
pred_df = pd.read_csv(upstream[upstream_name]["predictions"])
pred_df.head()

# %%
# assess Tanimoto similarity to training set
# load training data
train_df = pd.read_parquet(X_train_data)
similar_df = find_similar_train_ligand(train_df, pred_df)
similar_df.head()

# %%
# concat similar_df to pred_df
pred_sim_df = pd.concat([pred_df, similar_df], axis=1)
pred_sim_df = pred_sim_df.drop(columns=["Test SMILES"])

# %%
# flag misclassified examples in pred_sim_df
pred_sim_df["Correct"] = pred_sim_df["Class"] == pred_sim_df["Predicted value"]

# save df, for use with streamlit
pred_sim_df.to_csv(product["data"], index=False)

# %%
# some quick metrics
misclassified_df = pred_sim_df.loc[pred_sim_df["Correct"] == False]
log.info(f"Number of misclassified examples: {len(misclassified_df)}")
misclassified_df.head()
