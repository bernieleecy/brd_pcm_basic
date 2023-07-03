# %% [markdown]
# This file is to clean the data, and to write files with and without fingerprint
# duplicates

# %%
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from rdkit.DataManip.Metric import GetTanimotoSimMat

# %% tags=["parameters"]
upstream = None
product = None
brd_data = None
drop_sub_50 = None

# %%
# Read in the data from a specified csv file in the folder
# paths are relative to pipeline.yaml
data = pd.read_csv(brd_data, index_col=0)

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
data.sort_values(by=["Protein", "Class", "Type"], ascending=[True, False, False])

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

# %%
# Save the cleaned data to a csv file (this is prior to removing fingerprint duplicates)
data_2.to_csv(str(product["data"]))

# %%
# Now deal with the fingerprint duplicates
unique_ligands = data_2["Canon_SMILES"].unique()
n_ligands_all = len(unique_ligands)
print(f"Number of unique ligands: {n_ligands_all}")
# sanitize=True by default
ligand_mols = [Chem.MolFromSmiles(s) for s in unique_ligands]

# %%
# Now check for Tanimoto similarity 1 (at radius 3, nbits 1024, need chirality on)
all_morgan_fp_1024 = [
    AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=1024, useChirality=True)
    for m in ligand_mols
]
all_tanimoto_sim_1024 = GetTanimotoSimMat(all_morgan_fp_1024)

lig_idx = 0
index = 0
dups = []

for i in range(n_ligands_all):
    for j in range(i):
        if all_tanimoto_sim_1024[index] == 1:
            print(f"{index} Corresponds to ligand {lig_idx+1} against ligand {j+1}")
            dups.append(unique_ligands[i])
            dups.append(unique_ligands[j])
        index += 1
    lig_idx += 1

dups = set(dups)

# %%
# Get some information about the duplicates, then remove them
print(f"Number of duplicates: {len(dups)}")

data_no_dups = data_2.loc[~data_2["Canon_SMILES"].isin(dups)]
print(f"Remaining data points: {data_no_dups.shape[0]}")

# %%
# Save the data without duplicates to a csv file
data_no_dups.to_csv(str(product["data_no_dups"]))
