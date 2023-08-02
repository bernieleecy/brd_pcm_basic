# %% [markdown]
# This file is for examining the model performance on the test set

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.calibration import CalibrationDisplay

from importlib.resources import files
from brd_pcm.utils.evaluate import get_pr_auc, get_key_cmetrics

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
pr_auc = get_pr_auc(pred_df["Class"], pred_df["P (class 1)"])
log.info(f"PRAUC is {pr_auc:.3f}")
pr_disp = PrecisionRecallDisplay.from_predictions(
    pred_df["Class"], pred_df["P (class 1)"], ax=ax, color="black"
)

# change legend label name
pr_disp.line_.set_label(f"PRAUC = {pr_auc:.3f}")
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
# calculate confusion based metrics and save to csv
conf_metrics = get_key_cmetrics(
    y_true=pred_df["Class"],
    y_pred=pred_df["Predicted value"],
    y_pred_proba=pred_df["P (class 1)"],
)
conf_metrics_df = pd.DataFrame(conf_metrics, index=[0])

conf_metrics_df.to_csv(product["conf_metrics"], float_format="%.3f", index=False)
conf_metrics_df

# %%
# calibration curve
fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
calib_disp = CalibrationDisplay.from_predictions(
    pred_df["Class"], pred_df["P (class 1)"], n_bins=5, ax=ax, color="black"
)
ax.legend(frameon=False)

fig.savefig(product["cal_curve"], dpi=600)
