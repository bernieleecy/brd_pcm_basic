# %% [markdown]
# This file is for examining the model performance on the test set

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay

from pythia.pcm_tools import plot_by_protein_preds

# logging
import logging
logging.basicConfig(format='%(message)s')
log = logging.getLogger()
log.setLevel(logging.INFO)

# set plotting preferences
sns.set_style("ticks")
plt.style.use("plotstyle.mplstyle")
sns.set_palette("colorblind")

# %% tags=["parameters"]
upstream = ["train"]
product = None

# %%
# load data
clf_name = "Random Forest"
pred_df = pd.read_csv(upstream["train"]["predictions"])
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
roc_disp.line_.set_label(f"{clf_name} (AUC = {roc_auc:.3f})")
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
pr_disp.line_.set_label(f"{clf_name} (PRAUC = {prauc:.3f})")
ax.legend(frameon=False)
fig.savefig(product["pr_curve"], dpi=600)

# %%
conf_mat = confusion_matrix(pred_df["Class"], pred_df["Predicted value"], labels=(0, 1))
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["Inactive", "Active"])
disp.plot(cmap="rocket", ax=ax)
ax.set(xlabel="Predicted class", ylabel="Known class")

fig.savefig(product["cmat"], bbox_inches="tight", dpi=600)
