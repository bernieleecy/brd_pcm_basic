import pandas as pd
from importlib.resources import files

def validate_protein_names(product, params):
    """Check that protein names match expectations"""
    valid_file = files("brd_pcm.resources").joinpath("valid_proteins.csv")
    valid_proteins = pd.read_csv(valid_file,index_col=0).index.tolist()
    df = pd.read_csv(product["data"], index_col=0)
    protein_names = df["Protein"].unique()

    # Check that all protein names are in valid_proteins
    assert(all([p in valid_proteins for p in protein_names]))
