# %% [markdown]
# This file is for featurizing the data

# %%
from pathlib import Path
import pandas as pd
from molfeat.trans import MoleculeTransformer


def featurize_ligands(upstream, product):
    """Add features for the ligand
    Class information is not retained here"""
    upstream_name = list(upstream)[0]
    data = pd.read_csv(str(upstream[upstream_name]["data"]), index_col=0)

    # Initialise the ligand featurizer
    featurizer = "ecfp"
    lig_params = {"radius": 3, "nBits": 1024, "useChirality": True}

    # add features (this doesn't work if I store data.loc[:,"Canon_SMILES"] as a
    # variable, probably because it's a series)
    mol_transf = MoleculeTransformer(featurizer, **lig_params)
    lig_fps = mol_transf(data.loc[:, "Canon_SMILES"])
    lig_feats = pd.DataFrame(lig_fps).set_index(data.loc[:, "Canon_SMILES"])

    lig_feats.columns = lig_feats.columns.astype(str)
    # get Canon_SMILES as a column
    lig_feats = lig_feats.reset_index()

    # save lig_feats as parquet
    lig_feats.to_parquet(product["data"])


def featurize_proteins(upstream, product, protein_file):
    """Add features for the protein
    Class information is not retained here"""

    upstream_name = list(upstream)[0]
    protein_descriptor = Path(protein_file).stem
    data = pd.read_csv(str(upstream[upstream_name]["data"]), index_col=0)
    data_to_feat = data.loc[:, "Protein"]

    # check if tsv or csv file
    if protein_file.endswith(".tsv"):
        load_protein_df = pd.read_csv(protein_file, sep="\t", index_col=0)
    elif protein_file.endswith(".csv"):
        load_protein_df = pd.read_csv(protein_file, index_col=0)
    else:
        raise ValueError("Protein file must be either in tsv or csv format")

    # passing a set will be deprecated in the future, so convert to list
    protein_set = list(set(data_to_feat))
    relevant_proteins = load_protein_df.loc[protein_set, :]

    # merge data_to_feat and relevant_proteins, works because data_to_feat is a named
    # series
    protein_feats = pd.merge(
        data_to_feat, relevant_proteins, left_on="Protein", right_index=True
    )
    protein_feats = protein_feats.reset_index(drop=True)

    # save protein_feats as parquet file
    protein_feats.to_parquet(str(product["data"]))


def combine_features(upstream, product, known_classes):
    """Combine ligand and protein features
    Upstream is supplied as find_fp_dups, featurize_ligands, then featurize_protein
    """
    base_df = pd.read_csv(str(upstream["find_fp_dups"]["data"]))
    ligand_df = pd.read_parquet(str(upstream["featurize_ligands"]["data"]))
    protein_df = pd.read_parquet(str(upstream["featurize_proteins"]["data"]))

    # some massaging of data to aid inspection
    smiles_col = ligand_df.loc[:, "Canon_SMILES"]
    ligand_df = ligand_df.drop(columns=["Canon_SMILES"])
    protein_col = protein_df.loc[:, "Protein"]
    protein_df = protein_df.drop(columns=["Protein"])

    # merge the two dataframes
    merged_df = pd.concat([smiles_col, protein_col, ligand_df, protein_df], axis=1)

    if known_classes:
        merged_df["Class"] = base_df["Class"]

    print(merged_df.head())

    # save merged_df as parquet file
    merged_df.to_parquet(str(product["data"]))
