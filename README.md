# Ploomber pipelines for BRD PCM models

Using a ploomber pipeline to wrap the different steps of the BRD classifier work into a more reproducible format.

While developing this pipeline, dependencies had to be installed via conda-forge to get reproducibility (i.e. exact values that I obtained while prototyping).
Models were developed on a M1 mac mini, results may differ slightly on other architectures.

For exact reproducibility, use:
```
conda env create -n test_ploomber -f environment.dev.yml
```

Otherwise, use:
```
conda env create -n test_ploomber -f environment.yml
```

After installing the dependencies, the packaged pipeline **MUST** be installed with:
```
python -m pip install -e .
```

The Venn-ABERS calibration used the fast Venn-ABERS implementation from https://github.com/ptocca/VennABERS and https://github.com/valeman/Multi-class-probabilistic-classification/tree/main 

# Running pipelines

The two main pipelines used in the thesis were:
* `src/brd_pcm/pipeline.yaml` for model training
* `src/brd_pcm/pipeline.serve_fps.yaml` for serving predictions, starting from fingerprint–BRD pairs

Two other pipelines are included, use at your own risk:
* `src/brd_pcm/pipeline.serve.yaml` for serving predictions, starting from SMILES–BRD pairs
* `src/brd_pcm/pipeline.extra_test.yaml` for checking additional test sets

All the pipelines that use SMILES as input share the `pipeline.preprocessing.yaml` code.

Parameters for each pipeline are specified in the corresponding `env.yaml` or `env.{{type}}.yaml` file.

## Parameters for training pipeline 

The default parameters provided in `env.yaml` are:
```
root: "products/train"
brd_data: "inputs/chembl33_combined_init.csv"
protein_descriptor: "CKSAAGP"
drop_sub_50: True
known_classes: True
random_seed: 13579
```

The `root` and `brd_data` folders are specified relative to the main directory, `brd_pcm_basic`.
At this stage, the inputs folder is not supplied, for data privacy reasons. 
To run the pipeline with the default settings, enter the main directory, and run:
```
ploomber build
```

To change the input file used:
```
ploomber build --env-brd-data inputs/{{new_brd_data_file.csv}}
```

# Notes on data

The model training pipeline does some data cleaning (`clean.py` is a rewritten version of notebook 3 from the original data cleaning workflow).
As a result, the input file is expected to have the following columns: `["SMILES", "Protein", "Class"]`, and `clean.py` will check for these files.
Do not include Canon_SMILES as a column.

During model training, the order of the input data is NOT changed during cleaning and featurization
* Changing the order (e.g. reordering alphabetically by Protein name) can and will break the code
