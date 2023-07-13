# %% [markdown]
# This file is for an initial cleaning of the data, it does not remove duplicated
# fingerprints

# %%
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# %% tags=["parameters"]
upstream = None
product = None
brd_data = None
drop_sub_50 = None
known_classes = None

# %%
# Read in the data from a specified csv file in the folder
data = pd.read_csv(brd_data, index_col=0)

# %%
# Check for SMILES and Protein columns
if "SMILES" not in data.columns:
    raise ValueError("SMILES column not found")
if "Protein" not in data.columns:
    raise ValueError("Protein column not found")
# if classes is expected, check for the Class column
if known_classes:
    if "Class" not in data.columns:
        raise ValueError("Class column not found")

# %%
# Make Canon_SMILES column, then remove duplicates (prioritising ChEMBL entries)
canon_smiles = [Chem.CanonSmiles(s, useChiral=1) for s in data["SMILES"]]
data["Canon_SMILES"] = canon_smiles
data = data.drop_duplicates(subset=["Protein", "Canon_SMILES"], keep="first")

# %%
# Do data preparation and checks (based on https://github.com/vfscalfani/CSN_tutorial)
# First check for disconnected SMILES via string matching
data_2 = data[~data["Canon_SMILES"].str.contains("\.")].copy()

# Then double check for disconnected fragments and remove disconnected fragments
num_frags = []
for smi in data_2["Canon_SMILES"]:
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    num_frags.append(len(Chem.GetMolFrags(mol)))

data_2["num_frags"] = num_frags
data_2 = data_2[data_2["num_frags"] == 1]
# drop the num_frags column
data_2 = data_2.drop(columns=["num_frags"])

# %%
# Remove bromodomains with fewer than 50 entries (during training only)
if drop_sub_50:
    few_points = data_2["Protein"].value_counts() < 50  # boolean
    few_points_idx = few_points[few_points].index

    data_2 = data_2.loc[~data_2["Protein"].isin(few_points_idx)]

print(data_2.shape)
print(data_2["Canon_SMILES"].describe())
print(data_2["Protein"].describe())
print(f"Number of removed points: {data.shape[0] - data_2.shape[0]}")

# %%
# Save the cleaned data to a csv file (this is prior to removing fingerprint duplicates)
data_2.to_csv(str(product["data"]))
