# %%
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataManip.Metric import GetTanimotoSimMat

# %% tags=["parameters"]
upstream = None
product = None
known_classes = None

# %%
# Get the upstream name (assumes single upstream here)
upstream_name = list(upstream)[0]
# Load data
data = pd.read_csv(str(upstream[upstream_name]["data"]), index_col=0)

# %%
# Deal with the fingerprint duplicates
unique_ligands = data["Canon_SMILES"].unique()
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
duplicated_points = data.loc[data["Canon_SMILES"].isin(dups)]
print(f"Number of duplicate fingerprints: {len(dups)}")
print(f"Total duplicated points: {duplicated_points.shape[0]}")

# %%
# Save the fingerprint duplicates to enable further inspection
duplicated_points.to_csv(str(product["duplicates"]))

# %%
# Remove the duplicates from the data
data_no_dups = data.loc[~data["Canon_SMILES"].isin(dups)]
print(f"Remaining data points: {data_no_dups.shape[0]}")

# %%
# Save the data without duplicates to a csv file
data_no_dups.to_csv(str(product["data"]))
