# %% [markdown]
# This file is for test set analysis (individual proteins)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
log.info(f"Number of samples with BRD4-1: {len(pred_df)}")
log.info(f"Number of samples without BRD4-1: {len(no_brd4_1_df)}")

# %%
no_brd4_1_metrics = get_key_cmetrics(
    y_true=no_brd4_1_df["Class"],
    y_pred=no_brd4_1_df["Predicted value"],
    y_pred_proba=no_brd4_1_df["P (class 1)"],
)
conf_metrics_df = pd.DataFrame(no_brd4_1_metrics, index=[0])

conf_metrics_df

# %% [markdown]
# Now look at by family performance, loading in a dictionary of protein families
# Human bromodomains only, from Filippakopoulos et al. Cell 2012

# %% tags=["dictionary"]
brd_families = {
    "I": ["BPTF", "KAT2A", "PCAF", "CECR2"],
    "II": [
        "BRD4-1",
        "BRD4-2",
        "BRD2-1",
        "BRD2-2",
        "BRD3-1",
        "BRD3-2",
        "BRDT-1",
        "BRDT-2",
        "BAZ1A",
    ],
    "III": [
        "EP300",
        "CREBBP",
        "BRWD3-2",
        "PHIP-2",
        "BRD8-1",
        "BRD8-2",
        "BAZ1B",
        "WDR9-2",
    ],
    "IV": ["BRD1", "BRPF1A", "BRPF1B", "ATAD2", "BRD9", "BRD7", "BRPF3", "ATAD2B"],
    "V": ["TRIM66", "TRIM24", "SP140", "BAZ2B", "BAZ2A", "TRIM33A", "TRIM33B"],
    "VI": ["MLL", "TRIM28"],
    "VII": [
        "TAF1-1",
        "TAF1-2",
        "ZMYND11",
        "BRWD3-1",
        "TAF1L-1",
        "TAF1L-2",
        "PHIP-1",
        "PRKCBP1",
        "WDR9-1",
    ],
    "VIII": [
        "SMARCA2A",
        "SMARCA2B",
        "SMARCA4",
        "PB1-1",
        "PB1-2",
        "PB1-3",
        "PB1-4",
        "PB1-5",
        "PB1-6",
        "ASH1L",
    ],
}

# %%
metrics_dfs = []

for family, proteins in brd_families.items():
    check_df = pred_df.query("Protein in @proteins")
    if len(check_df) > 0:
        # only print out proteins that are actually in the dataset
        present_proteins = check_df["Protein"].unique()
        log.info(f"{family}: {present_proteins}")
        log.info(f"Number of samples: {len(check_df)}\n")
        # check metrics
        check_metrics = get_key_cmetrics(
            y_true=check_df["Class"],
            y_pred=check_df["Predicted value"],
            y_pred_proba=check_df["P (class 1)"],
        )
        check_metrics_df = pd.DataFrame(check_metrics, index=[0])
        check_metrics_df["Family"] = family
        metrics_dfs.append(check_metrics_df)

metrics_df = pd.concat(metrics_dfs, ignore_index=True)
metrics_df.set_index("Family", inplace=True)
metrics_df
