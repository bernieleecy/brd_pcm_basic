# %% [markdown]
# This file is for featurizing the data

# %%
import pandas as pd
from brd_pcm.pcm_tools.data_prep import AddFeatures

# %% tags=["parameters"]
upstream = None
product = None

# %%
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
# Load data and make separate dfs for ligand and protein features
# Possible Morgan fingerprint duplicates already removed here
data = pd.read_csv(str(upstream[upstream_name]["data_no_dups"]), index_col=0)
data = data[["Canon_SMILES", "Protein", "Class"]]

# %%
# Initialise the ligand featurizer
ligand_descriptor = "ecfp"
ligand_params = {"radius": 3, "nBits": 1024, "useChirality": True}
protein_descriptor = "CKSAAGP"
protein_file = f"protein_features/{protein_descriptor}.tsv"

# %%
# Initialise class for featurization
pcm_data = AddFeatures(
    data,
    smiles_col="Canon_SMILES",
    protein_col="Protein",
    class_col="Class",
)

# get protein and ligand features
pcm_data.get_protein_features(protein_file, name=protein_descriptor)
pcm_data.get_ligand_features_molfeat(
    ligand_descriptor, feature_path=None, **ligand_params
)
pcm_data.combine_feats()

# %%
# Save the data
pcm_data.pcm_feats_classes.to_parquet(product["data"])
