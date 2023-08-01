# %% [markdown]
# This file precedes model training
#
# It loads the data, splits it into train and test sets, and prepares the imblearn
# pipeline (stratified group split only!)

# %%
import pickle
import numpy as np
import pandas as pd

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

# imblearn
from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline

# custom functions
from brd_pcm.pcm_tools.data_prep import SplitData
from brd_pcm.pcm_tools.ml import make_imblearn_pipe, ros_by_protein_class

# logging
import logging

logging.basicConfig(format="%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)

# fix random seed for reproducibility, must be set here
random_seed = 13579
rng = np.random.RandomState(random_seed)
log.info(f"Random seed: {random_seed}")

# %% tags=["parameters"]
upstream = None
product = None

# %%
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
# Load data
all_data = pd.read_parquet(str(upstream[upstream_name]["data"]))
classes = all_data["Class"]
pcm_data = all_data.drop(columns=["Class"])

# %%
# split all_data into train and test (StratifiedGroupKFold, first fold here)
split_data = SplitData(pcm_data, classes)
split_data.stratified_group_split(
    test_size=0.2,
    group="murcko",
    smiles_col="Canon_SMILES",
    shuffle=True,
    random_state=random_seed,
)
print(split_data.check_disjoint())

# %%
# save the split data to parquet format
split_data.X_train.reset_index(drop=True).to_parquet(str(product["X_train"]))
split_data.X_test.reset_index(drop=True).to_parquet(str(product["X_test"]))
split_data.y_train.to_frame().reset_index(drop=True).to_parquet(str(product["y_train"]))
split_data.y_test.to_frame().reset_index(drop=True).to_parquet(str(product["y_test"]))

# %%
# get X_train to define ligand and prot feat names
X_train = split_data.X_train.drop(columns=["Canon_SMILES", "Murcko_SMILES"])
# define ligand and prot feat names (this is quite manual)
lig_feat_names = [str(i) for i in range(1024)]
# prot feat names are in df columns, but not in lig_feat_names
prot_feat_names = [
    col for col in X_train.columns if col not in [*lig_feat_names, "Protein"]
]

# %%
# setup the pipeline
clf = RandomForestClassifier(
    n_estimators=300, max_depth=15, random_state=random_seed, n_jobs=-1
)
sampler = FunctionSampler(
    func=ros_by_protein_class, validate=False, kw_args={"random_seed": random_seed}
)

pipe_clf = make_imblearn_pipe(
    train_lig_feats=lig_feat_names,
    train_prot_feats=prot_feat_names,
    sampler=sampler,
    train_clf=clf,
    var_thres=0.95,
    scaler=MinMaxScaler(),
)
pipe_clf.set_output(transform="pandas")

# %%
# pickle the pipeline
with open(product["imblearn_pipe"], "wb") as f:
    pickle.dump(pipe_clf, f)
