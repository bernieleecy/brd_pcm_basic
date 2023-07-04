# %% [markdown]
# This file is for examining the model performance on new predictions

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

import pythia.classification_metrics as cmetrics

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
conf_metrics_df = conf_metrics_df[
    [
        "matthews_correlation_coefficient",
        "precision",
        "recall",
        "tnr",
        "f1",
        "g-mean",
        "accuracy",
    ]
]

conf_metrics_df.to_csv(product["conf_metrics"], float_format="%.3f", index=False)
