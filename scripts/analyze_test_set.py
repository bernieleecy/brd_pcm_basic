# %% [markdown]
# This file is for examining the model performance on the test set

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay

import pythia.classification_metrics as cmetrics
from pythia.pcm_tools import get_by_protein_preds, plot_by_protein_preds

# logging
import logging

logging.basicConfig(format="%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)

# set plotting preferences
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
# get roc_auc score and curve
roc_auc = roc_auc_score(pred_df["Class"], pred_df["P (class 1)"])
log.info(f"roc_auc score is {roc_auc:.3f}")

fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
roc_disp = RocCurveDisplay.from_predictions(
    pred_df["Class"], pred_df["P (class 1)"], ax=ax, color="black"
)

# change legend label name
roc_disp.line_.set_label(f"AUC = {roc_auc:.3f}")
ax.legend(frameon=False)
fig.savefig(product["roc_curve"], dpi=600)

# %%
# get precision-recall curve, this doesn't really consider the inactive class
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
prec, rec, _ = precision_recall_curve(pred_df["Class"], pred_df["P (class 1)"])
prauc = auc(rec, prec)
log.info(f"PRAUC is {prauc:.3f}")
pr_disp = PrecisionRecallDisplay.from_predictions(
    pred_df["Class"], pred_df["P (class 1)"], ax=ax, color="black"
)

# change legend label name
pr_disp.line_.set_label(f"PRAUC = {prauc:.3f}")
ax.legend(frameon=False)
fig.savefig(product["pr_curve"], dpi=600)

# %%
# classification report
print(classification_report(pred_df["Class"], pred_df["Predicted value"]))

# %%
# confusion matrix
conf_mat = confusion_matrix(pred_df["Class"], pred_df["Predicted value"], labels=(0, 1))
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_mat, display_labels=["Inactive", "Active"]
)
disp.plot(cmap="rocket", ax=ax)
ax.set(xlabel="Predicted class", ylabel="Known class")

fig.savefig(product["cmat"], bbox_inches="tight", dpi=600)

# %%
# calculate confusion based metrics (this needs to be reworked) and save to csv
conf_metrics = cmetrics.calculate_confusion_based_metrics(conf_mat)
conf_metrics_df = pd.DataFrame(conf_metrics, index=[0])
conf_metrics_df["roc_auc"] = roc_auc
conf_metrics_df["prauc"] = prauc
conf_metrics_df = conf_metrics_df[
    [
        "matthews_correlation_coefficient",
        "roc_auc",
        "prauc",
        "precision",
        "recall",
        "tnr",
        "f1",
        "g-mean",
        "accuracy",
    ]
]

conf_metrics_df.to_csv(product["conf_metrics"], float_format="%.3f", index=False)

# %%
# calibration curve
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
calib_disp = CalibrationDisplay.from_predictions(
    pred_df["Class"], pred_df["P (class 1)"], n_bins=5, ax=ax, color="black"
)
ax.legend(frameon=False)

fig.savefig(product["cal_curve"], dpi=600)

# %%
# get results by protein (this also requires a rework)
indiv_prot_results = get_by_protein_preds(
    pred_df, protein_col="Protein", class_col="Class", outfile=None
)
indiv_prot_results.to_csv(product["indiv_prot_csv"], float_format="%.3f")

# %% plot all proteins on the same axes
fig, ax = plt.subplots(figsize=(13,5), constrained_layout=True)

plot_by_protein_preds(indiv_prot_results, ax=ax)

fig.savefig(product["indiv_prot_plot"], dpi=600)