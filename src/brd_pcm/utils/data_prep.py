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
        feature_file,
        sep="\t",
        index_col=0,
        name="protein_descriptor",
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
        self._protein_descriptor = name
        load_protein_df = pd.read_csv(feature_file, sep=sep, index_col=index_col)
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


class SplitData:
    """Custom class to handle different data splits
    For PCM binary classification, expects SMILES and protein data to be in X_data
    Relies heavily on pandas DataFrames!
    Important to make a copy of the data when initialising the class

    Attributes:
        X_data (pd DataFrame): DataFrame with shape (n_samples, n_cols).
        y_data (pd Series): Series with shape (n_samples, ).
        protein_class (pd Series): Series with shape (n_samples, ).
        length (int): Number of samples in X_data.
        X_train (pd DataFrame): DataFrame with shape (n_train, n_cols).
        X_test (pd DataFrame): DataFrame with shape (n_test, n_cols).
        y_train (pd Series): Series with shape (n_train, ).
        y_test (pd Series): Series with shape (n_test, ).
    """

    def __init__(self, X_data, y_data):
        """
        This works because y_data is a Series, not a DataFrame here
        """
        self.X_data = X_data.copy()
        self.y_data = y_data.copy()
        self.protein_class = (
            self.X_data["Protein"].values + "-" + self.y_data.astype(str)
        )
        self.length = len(X_data)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def __len__(self):
        return self.length

    def random_split(self, test_size=0.2, stratified_split=True, random_state=1):
        # With sklearn train_test_split, shuffles by default
        if stratified_split:
            stratify = self.protein_class
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_data,
            self.y_data,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )

    def stratified_group_split(
        self,
        test_size=0.2,
        group="murcko",
        smiles_col="Canon_SMILES",
        shuffle=False,
        random_state=1,
    ):
        """Do stratified group split
        Uses StratifiedGroupKFold and takes the first fold

        Args:
            test_size (float, optional): Used to calculate n_splits. Defaults
                to 0.2 (1/0.2 = 5 splits, 20% of groups in test).
            group (str, optional): Type of grouping to use. Defaults to
                "murcko", and only Murcko scaffolds are implemented at present.
            smiles_col (str, optional): Name of the column containing the
                canonical SMILES. Defaults to "Canon_SMILES"
            shuffle (bool, optional): Whether to shuffle. Defaults to False.
            random_state (int, optional): Random seed to use. Defaults to 1.
        """
        # return an integer for splits
        n_splits = int(1 / test_size)

        PandasTools.AddMoleculeColumnToFrame(
            self.X_data, smilesCol=smiles_col, molCol="ROMol"
        )
        PandasTools.AddMurckoToFrame(
            self.X_data, molCol="ROMol", MurckoCol="Murcko_SMILES"
        )
        # replace empty strings in murcko smiles col with nan
        self.X_data["Murcko_SMILES"].replace("", np.nan, inplace=True)
        self.X_data["Murcko_SMILES"].fillna("NoMurcko", inplace=True)
        self.X_data.drop("ROMol", axis=1, inplace=True)

        if shuffle:
            sgkf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
        else:
            sgkf = StratifiedGroupKFold(n_splits=n_splits)
        train_idx, test_idx = next(
            sgkf.split(self.X_data[smiles_col], self.protein_class, self.X_data["Murcko_SMILES"])
        )
        self.X_train = self.X_data.loc[train_idx]
        self.X_test = self.X_data.loc[test_idx]
        self.y_train = self.y_data.loc[train_idx]
        self.y_test = self.y_data.loc[test_idx]

    def check_disjoint(self, smiles_col="Canon_SMILES"):
        X_train_unique = set(self.X_train[smiles_col])
        X_test_unique = set(self.X_test[smiles_col])
        print(f"Unique ligands in train: {len(X_train_unique)}")
        print(f"Unique ligands in test: {len(X_test_unique)}")
        if X_train_unique.isdisjoint(X_test_unique):
            print("No overlap in ligands between train and test sets")
            return True
        else:
            overlap = X_train_unique.intersection(X_test_unique)
            print(f"Overlap between sets: {len(overlap)}")
            print(
                f"Percent of test set ligands in train: {len(overlap) / len(X_test_unique):.1%}"
            )
            return False
