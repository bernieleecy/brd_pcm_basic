# %% [markdown]
# This file is for test set analysis (individual proteins)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay

from brd_pcm.pcm_tools.evaluate import (
    get_key_cmetrics,
    get_by_protein_preds,
    plot_by_protein_preds,
)

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

# %%
# load data
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
pred_df = pd.read_csv(upstream[upstream_name]["predictions"])
pred_df.head()

# %%
# count number of samples per protein and class, and plot
fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)

sns.countplot(data=pred_df, x="Protein", hue="Class", ax=ax, edgecolor="black")
ax.set_ylabel("Count")
ax.tick_params(axis="x", which="both", labelsize=11, rotation=90)

for cont in ax.containers:
    ax.bar_label(cont, fontsize=8, padding=1)

_ = ax.legend(frameon=False, title="Class", loc="upper right")

# %%
# get results by protein
indiv_prot_results = get_by_protein_preds(
    pred_df, protein_col="Protein", class_col="Class", outfile=None
)
indiv_prot_results.to_csv(product["indiv_prot_csv"], float_format="%.3f")

# %% plot all proteins on the same axes
fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)

plot_by_protein_preds(indiv_prot_results, ax=ax)

fig.savefig(product["indiv_prot_plot"], dpi=600)


# %% [markdown]
# BRD4-1 makes up a disproportionate amount of the data, so check metrics without it

# %%
no_brd4_1_df = pred_df.query("Protein != 'BRD4-1'")
log.info(f'Number of samples with BRD4-1: {len(pred_df)}')
log.info(f"Number of samples without BRD4-1: {len(no_brd4_1_df)}")

# %%
no_brd4_1_metrics = get_key_cmetrics(
    y_true=no_brd4_1_df["Class"],
    y_pred=no_brd4_1_df["Predicted value"],
    y_pred_proba=no_brd4_1_df["P (class 1)"],
)
conf_metrics_df = pd.DataFrame(no_brd4_1_metrics, index=[0])

conf_metrics_df
