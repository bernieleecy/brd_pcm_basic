# for adding ligand and protein features

from pathlib import Path
import pandas as pd
from molfeat.trans import MoleculeTransformer
from importlib.resources import files

def get_protein_feats(data, protein_file):
    # file must be in .tsv format at the moment
    if str(protein_file).endswith(".tsv"):
        load_protein_df = pd.read_csv(protein_file, sep="\t", index_col=0)
    else:
        raise ValueError("Protein file must be either in tsv format")

    # passing a set will be deprecated in the future, so convert to list
    protein_set = list(set(data))
    relevant_proteins = load_protein_df.loc[protein_set, :]

    # merge data_to_feat and relevant_proteins, works because data_to_feat is a named
    # series
    # MUST state how="left", otherwise it gets sorted alphabetically and the
    # combine_feats function has unexpected behaviour
    protein_feats = pd.merge(
        data, relevant_proteins, how="left", left_on="Protein", right_index=True
    )
    protein_feats = protein_feats.reset_index(drop=True)
    return protein_feats


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


def featurize_proteins(upstream, product, protein_descriptor):
    """Add features for the protein
    Class information is not retained here"""

    upstream_name = list(upstream)[0]
    protein_file = files("brd_pcm.resources.brd_feats").joinpath(f"{protein_descriptor}.tsv")

    data = pd.read_csv(str(upstream[upstream_name]["data"]), index_col=0)
    data_to_feat = data.loc[:, "Protein"]

    # get protein features
    if protein_file.exists():
        protein_feats = get_protein_feats(data_to_feat, protein_file)
    else:
        raise FileNotFoundError("Protein descriptor file not found")

    # save protein_feats as parquet file
    protein_feats.to_parquet(str(product["data"]))


def combine_features(upstream, product, known_classes):
    """Combine ligand and protein features
    Upstream is supplied as find_fp_dups, featurize_ligands, then featurize_protein
    Expects parquet files for ligand and protein features
    """
    base_df = pd.read_csv(str(upstream["find_fp_dups"]["data"]))
    ligand_df = pd.read_parquet(str(upstream["featurize_ligands"]["data"]))
    protein_df = pd.read_parquet(str(upstream["featurize_proteins"]["data"]))

    # check that the order is the same in base_df and the new_dfs
    if not base_df["Canon_SMILES"].equals(ligand_df["Canon_SMILES"]):
        raise ValueError(
            "Order of ligand features was not retained during featurization"
        )
    if not base_df["Protein"].equals(protein_df["Protein"]):
        raise ValueError(
            "Order of protein features was not retained during featurization"
        )

    # some massaging of data to aid inspection
    smiles_col = ligand_df.loc[:, "Canon_SMILES"]
    ligand_df = ligand_df.drop(columns=["Canon_SMILES"])
    protein_col = protein_df.loc[:, "Protein"]
    protein_df = protein_df.drop(columns=["Protein"])

    merged_df = pd.concat([smiles_col, protein_col], axis=1)
    if known_classes:
        merged_df["Class"] = base_df["Class"].values
    merged_df = pd.concat([merged_df, ligand_df, protein_df], axis=1)

    print(merged_df.head())

    # save merged_df as parquet file
    merged_df.to_parquet(str(product["data"]))


def enter_from_fps(product, fp_data, protein_descriptor):
    """For entering a pipeline starting from ligand fps + protein names
    Only applicable during model serving"""

    data = pd.read_parquet(fp_data)

    # check for Protein column
    if "Protein" not in data.columns:
        raise ValueError("Protein column not found in fp_data")

    data_to_feat = data.loc[:, "Protein"]

    # get protein features
    protein_file = files("brd_pcm.resources.brd_feats").joinpath(f"{protein_descriptor}.tsv")
    if protein_file.exists():
        protein_feats = get_protein_feats(data_to_feat, protein_file)
    else:
        raise FileNotFoundError("Protein descriptor file not found")

    # combine with ligand feats now
    if not data["Protein"].equals(protein_feats["Protein"]):
        raise ValueError(
            "Order of protein features was not retained during featurization"
        )
    # combine all except the Protein column from protein_feats
    protein_feats = protein_feats.drop(columns=["Protein"])
    merged_df = pd.concat([data, protein_feats], axis=1)

    # save merged_df as parquet file
    merged_df.to_parquet(str(product["data"]))
