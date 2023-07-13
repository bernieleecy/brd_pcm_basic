import pandas as pd
import pkgutil

def validate_protein_names(product, params):
    """Check that protein names match expectations"""
    valid_proteins = pd.read_csv(params["protein_file"], sep="\t", index_col=0).index.tolist()
    df = pd.read_csv(product["data"], index_col=0)
    protein_names = df["Protein"].unique()

    # Check that all protein names are in valid_proteins
    assert(all([p in valid_proteins for p in protein_names]))
