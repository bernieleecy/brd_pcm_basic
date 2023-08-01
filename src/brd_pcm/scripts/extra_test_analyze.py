# %% [markdown]
# This file is for examining the model performance on unseen ligands with known classes

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from brd_pcm.utils.evaluate import get_key_cmetrics

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
conf_metrics = get_key_cmetrics(
    y_true=pred_df["Class"],
    y_pred=pred_df["Predicted value"],
    y_pred_proba=pred_df["P (class 1)"],
)
conf_metrics_df = pd.DataFrame(conf_metrics, index=[0])

conf_metrics_df.to_csv(product["conf_metrics"], float_format="%.3f", index=False)
conf_metrics_df
