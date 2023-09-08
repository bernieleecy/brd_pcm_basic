# for adding ligand and protein features
import os
from pathlib import Path
import pandas as pd
from molfeat.trans import MoleculeTransformer
from importlib.resources import files


def load_protein_feats(protein_descriptor, index_col=0):
    """Load protein features from .tsv or .csv file

    Args:
        protein_descriptor (str): Name of protein descriptor.
        index_col (int, optional): Index column for pd.read_csv. Defaults to 0.

    Returns:
        pd DataFrame: DataFrame with shape (n_samples, n_cols).
    """
    protein_descriptor = protein_descriptor.upper()
    base_path = files("brd_pcm.resources.brd_feats")

    # Check for .tsv file
    feature_file = base_path.joinpath(f"{protein_descriptor}.tsv")
    if not os.path.exists(feature_file):
        # Check for .csv file
        feature_file = base_path.joinpath(f"{protein_descriptor}.csv")
        if not os.path.exists(feature_file):
            raise FileNotFoundError(
                f"No .tsv or .csv file found for {protein_descriptor}"
            )

    # Load the file based on its extension
    if feature_file.suffix == ".tsv":
        load_protein_df = pd.read_csv(feature_file, sep="\t", index_col=index_col)
    elif feature_file.suffix == ".csv":
        load_protein_df = pd.read_csv(feature_file, index_col=index_col)
    else:
        raise ValueError("Protein features file needs to be .tsv or .csv")

    return load_protein_df


def get_protein_feats(data, protein_descriptor):
    """Get protein features for a given set of proteins"""
    load_protein_df = load_protein_feats(protein_descriptor, index_col=0)

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

    data = pd.read_csv(str(upstream[upstream_name]["data"]), index_col=0)
    data_to_feat = data.loc[:, "Protein"]
    protein_feats = get_protein_feats(data_to_feat, protein_descriptor)

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
    Only applicable during model serving

    Args:
        product (dict): product files to be produced.
        fp_data (str): path to parquet file containing ligand fps.
        protein_descriptor (str): protein descriptor name.
    """

    data = pd.read_parquet(fp_data)

    # check for Protein column
    if "Protein" not in data.columns:
        raise ValueError("Protein column not found in fp_data")

    data_to_feat = data.loc[:, "Protein"]

    # load protein features
    protein_feats = get_protein_feats(data_to_feat, protein_descriptor)

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
