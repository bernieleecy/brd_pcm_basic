# %% [markdown]
# This file is for featurizing the data

# %%
from pathlib import Path
import pandas as pd
from brd_pcm.pcm_tools.data_prep import AddFeatures

# %% tags=["parameters"]
upstream = None
product = None
known_classes = None
protein_file = None

# %%
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
# Load data and make separate dfs for ligand and protein features
# Possible Morgan fingerprint duplicates already removed here
data = pd.read_csv(str(upstream[upstream_name]["data"]), index_col=0)
data_to_feat = data.loc[:,["Canon_SMILES", "Protein"]]

# %%
# Initialise the ligand featurizer
ligand_descriptor = "ecfp"
ligand_params = {"radius": 3, "nBits": 1024, "useChirality": True}
protein_descriptor = Path(protein_file).stem

# %%
# Initialise class for featurization
pcm_data = AddFeatures(
    data_to_feat,
    smiles_col="Canon_SMILES",
    protein_col="Protein",
)

# get protein and ligand features
pcm_data.get_protein_features(protein_file, name=protein_descriptor)
pcm_data.get_ligand_features_molfeat(
    ligand_descriptor, feature_path=None, **ligand_params
)
pcm_data.combine_feats()
pcm_data.pcm_feats.head()

# %%
# add classes to the dataframe if needed
if known_classes:
    pcm_data.pcm_feats["Class"] = data["Class"].values

# %%
# Save the data
pcm_data.pcm_feats.to_parquet(product["data"])
