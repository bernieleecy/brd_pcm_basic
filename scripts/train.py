# %% [markdown]
# This file is for training the model (uncalibrated model first)

# %%
import numpy as np
import pandas as pd

# sklearn
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import column_transformer
from sklearn.ensemble import RandomForestClassifier

# imblearn
from imblearn import FunctionSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

# custom pythia functions
from pythia.pcm_tools import SplitData

# logging
import logging
logging.basicConfig(format='%(message)s')
log = logging.getLogger()
log.setLevel(logging.INFO)

# fix random seed for reproducibility
random_seed = 13579
rng = np.random.RandomState(random_seed)
log.info(f"Random seed: {random_seed}")

# %% tags=[parameters]
upstream = ["featurize"]
product = None

# %%
# load data
all_data = pd.read_parquet(str(upstream["featurize"]["data"]))


