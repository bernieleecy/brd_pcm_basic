# %% [markdown]
# This file is for processing predictions from new ligand fingerprints
# Assumes that ligands provided are unique
# Further processing is in a separate script, because running this script is slow

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from brd_pcm.utils.evaluate import get_key_cmetrics, find_similar_train_ligand_fps

# logging
import logging

logging.basicConfig(format="%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)

# set display and plotting preferences
pd.options.display.float_format = "{:.3f}".format
sns.set_style("ticks")
plt.style.use("plotstyle.mplstyle")
sns.set_palette("colorblind")

# %% tags=["parameters"]
upstream = None
product = None
X_train_data = None
X_test_data = None
fp_data = None

# %%
# load data
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
pred_df = pd.read_parquet(upstream[upstream_name]["predictions"])
pred_df.head()

# %%
# count predicted values
counts = pred_df["Predicted value"].value_counts()
log.info(f"Predicted values:\n{counts}")

# %%
# assess Tanimoto similarity to training set
# load training data
train_df = pd.read_parquet(X_train_data)
# load fingerprints data (100k ligands uses almost 1GB of memory, so be careful)
fp_df = pd.read_parquet(fp_data)
similar_train_df = find_similar_train_ligand_fps(train_df, fp_df)
similar_train_df.head()

# %%
# repeat on the test set
test_df = pd.read_parquet(X_test_data)
similar_test_df = find_similar_train_ligand_fps(test_df, fp_df)
similar_test_df = similar_test_df.rename(
    columns={
        "Closest Train SMILES": "Closest Test SMILES",
        "Tanimoto Similarity": "Tanimoto Similarity (Test)",
    }
)

# %%
# combine similar_dfs and pred_df
combined_df = pd.concat([pred_df, similar_train_df, similar_test_df], axis=1)
combined_df.head()

# %%
# save df
combined_df.to_parquet(product["similar_df"])
