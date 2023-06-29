# %% [markdown]
# This file is for cleaning the data

# %%
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools

# %% tags=["parameters"]
upstream = None
product = None

# %%
# Read in the data from a specified csv file in the folder
# paths are relative to pipeline.yaml
data = pd.read_csv("data/chembl33_combined_init.csv", index_col=0)

# %%
# Check for SMILES, Protein and Class columns (all must be present)
if "SMILES" not in data.columns:
    raise ValueError("SMILES column not found")
if "Protein" not in data.columns:
    raise ValueError("Protein column not found")
if "Class" not in data.columns:
    raise ValueError("Class column not found")

# %%
# Make Canon_SMILES column, then remove duplicates (prioritising ChEMBL entries)
canon_smiles = [Chem.CanonSmiles(s, useChiral=1) for s in data["SMILES"]]
data["Canon_SMILES"] = canon_smiles
data = data.drop_duplicates(subset=["Protein", "Canon_SMILES"], keep="first")

# %%
# Sort data in the desired order (requires conversion of strings to uppercase first)
data["Type"] = data["Type"].str.upper()
data.sort_values(by=["Protein","Class","Type"], ascending=[True,False,False])

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
# Remove bromodomains with fewer than 50 entries
few_points = (data_2["Protein"].value_counts() < 50) # boolean
few_points_idx = few_points[few_points].index

data_2 = data_2.loc[~data_2["Protein"].isin(few_points_idx)]

print(data_2.shape)
print(data_2["Canon_SMILES"].describe())
print(data_2["Protein"].describe())

# %%
# Add Murcko SMILES column with PandasTools
PandasTools.AddMoleculeColumnToFrame(data_2, "Canon_SMILES", "Mol")
PandasTools.AddMurckoToFrame(data_2, molCol="Mol", MurckoCol="Murcko_SMILES")

# replace empty strings in murcko smiles col with nan, then assign to NoMurcko
data_2["Murcko_SMILES"] = data_2["Murcko_SMILES"].replace("", np.nan)
data_2["Murcko_SMILES"] = data_2["Murcko_SMILES"].fillna("NoMurcko")
data_2.drop(columns=["Mol"], inplace=True)

# %%
# Save the cleaned data to a csv file (this is prior to removing fingerprint duplicates)
data_2.to_csv(str(product["data"]))
