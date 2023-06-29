# %% [markdown]
# This file is for featurizing the data

# %%
import pandas as pd
from pythia.pcm_tools import AddFeatures

# %% tags=["parameters"]
upstream = ["clean"]
product = None

# %%
# Load data and make separate dfs for ligand and protein features
data = pd.read_csv(str(upstream["clean"]["data"]), index_col=0)
data = data[["Canon_SMILES", "Protein", "Murcko_SMILES", "Class"]]

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
    group_col="Murcko_SMILES",
)

# get protein and ligand features
pcm_data.get_protein_features(protein_file, name=protein_descriptor)
pcm_data.get_ligand_features_molfeat(ligand_descriptor, feature_path=None, **ligand_params)

# retain all information as this will be needed to do train-test splits!
pcm_data.combine_feats(file_out=False, drop=False)

# %%
# Save the data
pcm_data.pcm_feats_classes.to_parquet(product["data"])
