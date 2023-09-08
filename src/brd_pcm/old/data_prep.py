# data preparation for PCM modelling
# has a class for adding features and a class for splitting data
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from rdkit import Chem
from rdkit.Chem import PandasTools
from molfeat.trans import MoleculeTransformer

import logging

log = logging.getLogger(__name__)

import os
import pandas as pd

def load_protein_feats(protein_descriptor, index_col=0):
    """ Load protein features from .tsv or .csv file

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
            raise FileNotFoundError(f"No .tsv or .csv file found for {protein_descriptor}")

    # Load the file based on its extension
    if feature_file.suffix == ".tsv":
        load_protein_df = pd.read_csv(feature_file, sep="\t", index_col=index_col)
    elif feature_file.suffix == ".csv":
        load_protein_df = pd.read_csv(feature_file, index_col=index_col)
    else:
        raise ValueError("Protein features file needs to be .tsv or .csv")

    return load_protein_df

class AddFeatures:

    """A class to add features to protein and ligands prior to splitting datasets.
    Only takes in ligand and protein information.

    Attributes:
        df (pd DataFrame): DataFrame with shape (n_samples, n_cols).
        smiles_col (str): Name of column containing SMILES strings. Defaults to "Canon_SMILES".
        protein_col (str): Name of column containing protein names. Defaults to "Protein".
        pcm_feats_classes (pd DataFrame): DataFrame with shape (n_samples,
            n_cols+n_feats). Produced by combine_feats().
    """

    def __init__(
        self,
        df,
        smiles_col="Canon_SMILES",
        protein_col="Protein",
    ):
        self.df = df.copy()
        self.smiles_col = smiles_col
        self.protein_col = protein_col
        self.pcm_feats = None

    def get_ligand_features_molfeat(self, featurizer, feature_path=None, **params):
        """Use the molfeat package to featurise ligands

        Args:
            featurizer (str): the name of the featurizer to use
            feature_path (str, optional): the path to save the yaml file to. Defaults to None.
        """
        mol_transf = MoleculeTransformer(featurizer, **params)
        tmp_lig_feats = mol_transf(self.df[self.smiles_col])
        self._lig_feats = pd.DataFrame(tmp_lig_feats)

        self._lig_feats = self._lig_feats.set_index(
            [
                self.df[self.smiles_col],
                self.df[self.protein_col],
            ]
        )

        # convert columns to str to help w/ sklearn later on
        self._lig_feats.columns = self._lig_feats.columns.astype(str)

        # save yaml file if desired
        if feature_path is not None:
            if not os.path.exists(feature_path):
                os.makedirs(feature_path)
            outfile = os.path.join(feature_path, f"{featurizer}.yaml")
            mol_transf.to_state_yaml_file(outfile)

        log.info(f"Obtained {featurizer} features")

    def get_protein_features(
        self,
        protein_descriptor,
        index_col=0,
    ):
        """Get features for the proteins
        Requires a protein features file to be supplied, be careful to select
        the relevant proteins only.
        Assumes that features are continuous variables.

        Args:
            feature_file (str): Path to file.
            sep (str, optional): File separator for pd.read_csv. Defaults to "\t".
            index_col (int, optional): Index column for pd.read_csv. Defaults to 0.
            name (str, optional): Protein descriptor name. Defaults to
                "protein_descriptor".
        """
        self._protein_descriptor = protein_descriptor
        load_protein_df = load_protein_feats(protein_descriptor, index_col=index_col)
        protein_set = set(self.df[self.protein_col])

        if protein_set.issubset(load_protein_df.index):
            log.info(f"Proteins are present in protein features file")
        else:
            raise ValueError("Proteins need to be in the protein features file")
        self._protein_df = load_protein_df.loc[load_protein_df.index.isin(protein_set)]

        log.info(f"Obtained protein {name} feature from file")

    def combine_feats(self):
        """Combines ligand and protein features.
        Meging with how="left" is critical for preserving order

        Returns:
            pd DataFrame: DataFrame with shape (n_samples, n_cols).
        """
        self.pcm_feats = self._lig_feats.reset_index().merge(
            self._protein_df, how="left", left_on="Protein", right_index=True
        )
