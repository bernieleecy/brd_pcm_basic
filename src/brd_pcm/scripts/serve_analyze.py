# %% [markdown]
# This file is for examining the model performance on unseen ligands with unknown
# classes

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from brd_pcm.utils.evaluate import get_key_cmetrics, find_similar_train_ligand

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

# %%
# load data
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
pred_df = pd.read_csv(upstream[upstream_name]["predictions"])
pred_df.head()

# %%
# count predicted values
counts = pred_df["Predicted value"].value_counts()
log.info(f"Predicted values:\n{counts}")

# %%
# assess Tanimoto similarity to training set
# load training data
train_df = pd.read_parquet(X_train_data)
similar_df = find_similar_train_ligand(train_df, pred_df)
similar_df.head()

# %%
# flag ligands that are also present in the training set (Tanimoto similarity 1)
similar_df_no_dup = similar_df.drop_duplicates(subset=["Test SMILES"])
ligands_in_train = similar_df_no_dup.query("`Tanimoto Similarity` == 1")
log.info(f"Unique ligands: {len(similar_df_no_dup)}")
log.info(f"Ligands in training set:{len(ligands_in_train)}")


# %%
# check predictions for ligands that are also in training
overlap_train_df = pred_df.query("`Canon_SMILES` in @ligands_in_train['Test SMILES']")
overlap_train_df.head()

# %%
# combine similar_df and pred_df
combined_df = pd.concat([pred_df, similar_df], axis=1)
combined_df = combined_df.drop(columns=["Test SMILES"])
combined_df.head()

# %%
# plot p0/p1 discordance against P (class 1)
fig, ax = plt.subplots(figsize=(5, 4))
sns.scatterplot(data=combined_df, x="P (class 1)", y="diff_p1_p0", ax=ax)
ax.set(xlabel="P (class 1)", ylabel="p1 - p0", xlim=(-0.05, 1.05), ylim=(0, 0.10))

# %%
# plot p0/p1 discordance against Tanimoto similarity of nearest neighbour
fig, ax = plt.subplots(figsize=(5, 4))
sns.scatterplot(data=combined_df, x="Tanimoto Similarity", y="diff_p1_p0", ax=ax)
ax.set(
    xlabel="Tanimoto Similarity", ylabel="p1 - p0", xlim=(-0.05, 1.05), ylim=(0, 0.10)
)

# %%
# output the combined df with Tanimoto similarity of closest neighbour
combined_df.to_csv(product["similar_df"], index=False)
