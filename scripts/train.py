# %% [markdown]
# This file is for training the model without calibration

# %%
import pickle
import numpy as np
import pandas as pd

# sklearn
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer, recall_score

# imblearn
from imblearn import FunctionSampler

# custom functions
from pythia.pcm_tools import SplitData
from functions.custom_pipeline import make_imblearn_pipe, ros_by_protein_class

# logging
import logging

logging.basicConfig(format="%(message)s")
log = logging.getLogger()
log.setLevel(logging.INFO)

# fix random seed for reproducibility
random_seed = 13579
rng = np.random.RandomState(random_seed)
log.info(f"Random seed: {random_seed}")

# %% tags=["parameters"]
upstream = ["featurize"]
product = None

# %%
# load data
all_data = pd.read_parquet(str(upstream["featurize"]["data"]))
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
# store X_train groups, then remove Canon_SMILES and Murcko_SMILES (Protein needed for
# oversampling)
train_groups = split_data.X_train["Murcko_SMILES"]
X_train = split_data.X_train.drop(columns=["Canon_SMILES", "Murcko_SMILES"])
y_train = split_data.y_train
X_test = split_data.X_test.drop(columns=["Canon_SMILES", "Murcko_SMILES"])
y_test = split_data.y_test

# %%
# define ligand and prot feat names (this is quite manual)
lig_feat_names = [str(i) for i in range(1024)]
# prot feat names are in df columns, but not in lig_feat_names
prot_feat_names = [
    col for col in X_train.columns if col not in [*lig_feat_names, "Protein"]
]

# %%
# setup the pipeline
cross_val_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_seed)
cross_val_clf = RandomForestClassifier(
    n_estimators=300, max_depth=15, random_state=random_seed, n_jobs=-1
)
sampler = FunctionSampler(
    func=ros_by_protein_class, validate=False, kw_args={"random_seed": random_seed}
)

pipe_clf = make_imblearn_pipe(
    train_lig_feats=lig_feat_names,
    train_prot_feats=prot_feat_names,
    sampler=sampler,
    train_clf=cross_val_clf,
    var_thres=0.95,
    scaler=MinMaxScaler(),
)
pipe_clf.set_output(transform="pandas")

# %%
# Check cross-validation scores
scoring = {
    "roc_auc": "roc_auc",
    "sensitivity": make_scorer(recall_score, pos_label=1),
    "specificity": make_scorer(recall_score, pos_label=0),
}

cross_val_scores = cross_validate(
    pipe_clf,
    X=X_train,
    y=y_train,
    groups=train_groups,
    cv=cross_val_cv,
    scoring=scoring,
)

# %%
# save cross_val_scores
cross_val_scores_df = pd.DataFrame(cross_val_scores)
mean_scores = cross_val_scores_df.mean()
std_scores = cross_val_scores_df.std()
cross_val_scores_df.loc["mean"] = mean_scores
cross_val_scores_df.loc["std"] = std_scores
# transpose df before saving
cross_val_scores_df.T.to_csv(product["cross_val"], index=False)

# %%
# Fit model on all training data
pipe_clf.fit(X_train, y_train)
# pickle the pipeline (where the RF model is still uncalibrated)
with open(product["model"], "wb") as f:
    pickle.dump(pipe_clf, f)

# %%
# Predict on test data
y_pred = pipe_clf.predict(X_test)
y_pred_proba = pipe_clf.predict_proba(X_test)

# %%
# save predictions
pred_df = split_data.X_test[["Canon_SMILES", "Protein", "Murcko_SMILES"]]
pred_df["Class"] = y_test
pred_df["Predicted value"] = y_pred
pred_df["P (class 0)"] = y_pred_proba[:, 0]
pred_df["P (class 1)"] = y_pred_proba[:, 1]

pred_df.to_csv(product["predictions"], index=False)
