# %% [markdown]
"""
This file is for selecting ligands of interest from screening

Assumes that ligands provided are unique
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from importlib.resources import files

# logging
import logging

logging.basicConfig(format="%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)

# set display and plotting preferences
pd.options.display.float_format = "{:.3f}".format
sns.set_style("ticks")
plotstyle = files("brd_pcm.resources").joinpath("plotstyle.mplstyle")
plt.style.use(plotstyle)
sns.set_palette("colorblind")

# %% tags=["parameters"]
upstream = None
product = None
X_train_data = None
y_train_data = None
X_test_data = None
y_test_data = None
frac_to_screen = None

# %%
# load predicted data
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
pred_df = pd.read_parquet(upstream[upstream_name]["similar_df"])

# %%
# load training and test data
X_train = pd.read_parquet(
    X_train_data, columns=["Canon_SMILES", "Protein", "Murcko_SMILES"]
)
y_train = pd.read_parquet(y_train_data)
train_df = pd.concat([X_train, y_train], axis=1)

X_test = pd.read_parquet(
    X_test_data, columns=["Canon_SMILES", "Protein", "Murcko_SMILES"]
)
y_test = pd.read_parquet(y_test_data)
test_df = pd.concat([X_test, y_test], axis=1)

# %%
# count predicted values (again)
counts = pred_df["Predicted value"].value_counts()
log.info(f"Predicted values:\n{counts}")

# %% [markdown]
"""
# Detecting overlaps

Looking for overlaps between the library and the training/test sets
"""

# %%
overlap_ids = []

# %%
# find ligands with Tanimoto similarity = 1 to training set
df_train_overlap = pred_df.query("`Tanimoto Similarity` == 1")
log.info(
    f"Number of ligands with Tanimoto similarity = 1 to training set: {len(df_train_overlap)}"
)
df_train_overlap.head(10)

# %%
# if there are overlapping compounds, then check for compound-target pair overlap
if len(df_train_overlap) > 0:
    df_full_train_overlap = df_train_overlap.merge(
        train_df,
        left_on=["Closest Train SMILES", "Protein"],
        right_on=["Canon_SMILES", "Protein"],
        how="inner",
    )
    if len(df_full_train_overlap) > 0:
        running_nos = df_full_train_overlap["Running number"].values
        overlap_ids.extend(running_nos)
        log.info("Compound-target pair overlap detected!")
        log.info(f"Running numbers: {running_nos}")
    else:
        log.info(
            "Some compound overlap detected, but there is no compound-target pair overlap"
        )
else:
    log.info("No compound overlap detected between library and training set")

# %%
# find ligands with Tanimoto similarity = 1 to training set
df_test_overlap = pred_df.query("`Tanimoto Similarity (Test)` == 1")
log.info(
    f"Number of ligands with Tanimoto similarity = 1 to test set: {len(df_test_overlap)}"
)
df_test_overlap.head(10)

# %%
# if there are overlapping compounds, then check for compound-target pair overlap
if len(df_test_overlap) > 0:
    df_full_test_overlap = df_test_overlap.merge(
        test_df,
        left_on=["Closest Test SMILES", "Protein"],
        right_on=["Canon_SMILES", "Protein"],
        how="inner",
    )
    if len(df_full_test_overlap) > 0:
        running_nos = df_full_test_overlap["Running number"].values
        log.info("Compound-target pair overlap detected!")
        log.info(f"Running numbers: {running_nos}")
        overlap_ids.extend(running_nos)
    else:
        log.info(
            "Some compound overlap detected, but there is no compound-target pair overlap"
        )
else:
    log.info("No compound overlap detected between library and test set")

# %%
# remove compound-target overlaps from the pred_df
log.info(f"Removing the following running numbers: {overlap_ids}")
pred_df = pred_df[~pred_df["Running number"].isin(overlap_ids)]

log.info(len(pred_df))


# %% [markdown]
"""
# Selecting ligands for screening

Still in the preliminary stages here!
"""
# %%
num_to_screen = int(len(pred_df) * frac_to_screen)
# and make sure it is an even number
num_to_screen = num_to_screen + (num_to_screen % 2)
max_each_class = num_to_screen // 2  # returns an integer
log.info(f"Number of ligands to select: {num_to_screen}")

# %%
# Find all active ligands, select all of them if there are fewer than max_each_class of
# them
active_ids = []
active_df = pred_df.query("`Predicted value` == 1")
log.info(f"Number of predicted actives: {len(active_df)}")

if len(active_df) < max_each_class:
    log.info("Selecting all predicted actives")
    active_ids = active_df["Running number"].values
else:
    # sort by Tanimoto similarity to train
    active_df = active_df.sort_values(by="Tanimoto Similarity", ascending=True)
    # select up to max_each_class
    active_df = active_df.iloc[:max_each_class, :]
    active_ids = active_df["Running number"].values

# %%
log.info(f"Number of active ligands selected: {len(active_ids)}")
log.info(f"Selected actives: {active_ids}")
log.info(f"Mean Tanimoto similarity: {active_df['Tanimoto Similarity'].mean():.3f}")

# %%
# For inactives, go down P (class 1) from <= 0.5 to <= 0.1
# Then "fill up" the max_each_class value
# active threshold is 0.5
n_ligs = 0
inactive_df = pd.DataFrame()
inactive_ids = []

for upper_limit in reversed(np.linspace(0.1, 0.5, 5)):
    lower_limit = upper_limit - 0.1
    log.info(f"Lower limit: {lower_limit:.1f}, Upper limit: {upper_limit:.1f}")
    inactives_at_thres = pred_df.query(
        "`P (class 1)` > @lower_limit & `P (class 1)` <= @upper_limit"
    )
    log.info(
        f"Number of predicted inactives at {lower_limit:.1f} to {upper_limit:.1f}: {len(inactives_at_thres)}"
    )
    n_ligs += len(inactives_at_thres)
    if n_ligs >= max_each_class:
        exceed_amount = n_ligs - max_each_class
        log.info(f"Max inactive ligs exceeded by: {exceed_amount}")
        n_to_keep = len(inactives_at_thres) - exceed_amount
        log.info(f"Number of inactives to keep: {n_to_keep}")
        # prioritise ligands closer to the decision boundary
        inactives_at_thres = (
            inactives_at_thres
            .sort_values(by="P (class 1)", ascending=False)
            .iloc[:n_to_keep, :]
        )
        inactive_df = pd.concat([inactive_df, inactives_at_thres])
        inactive_ids.extend(inactives_at_thres["Running number"].values)
        break
    else:
        inactive_df = pd.concat([inactive_df, inactives_at_thres])
        inactive_ids.extend(inactives_at_thres["Running number"].values)

# %%
log.info(f"Number of inactive ligands selected: {len(inactive_ids)}")
log.info(f"Selected inactives: {inactive_ids}")
log.info(f"Mean Tanimoto similarity: {inactive_df['Tanimoto Similarity'].mean():.3f}")

# %%
# combine the active and inactives and write to a file
proposed_all_df = pd.concat([active_df, inactive_df])
proposed_all_df = proposed_all_df.sort_values(by="P (class 1)", ascending=False)
proposed_all_df.to_csv(product["proposed"], index=False)

# %%
# write actives and inactives to a file
with open(product["summary"], "w") as f:
    f.write("# Proposed ligands for screening\n\n")
    f.write(f"Selected {len(active_ids)} active ligands\n")
    f.write("Active ligand IDs as a list:\n")
    f.write("".join([str(x) + ", " for x in active_ids]))
    f.write("\n")

    f.write("\n")
    f.write(f"Selected {len(inactive_ids)} inactive ligands\n")
    f.write("Inactive ligand IDs as a list:\n")
    f.write("".join([str(x) + ", " for x in inactive_ids]))
    f.write("\n")

    f.write("\n")
    f.write(f"Number of compound-target overlaps: {len(overlap_ids)}\n")
    f.write("Compound-target overlap IDs as a list:\n")
    f.write("".join([str(x) + ", " for x in overlap_ids]))
    f.write("\n")
    f.write("These compounds are not included in the proposed set\n")
