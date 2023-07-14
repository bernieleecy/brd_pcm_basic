# My first ploomber pipeline

Using a ploomber pipeline to wrap the different steps of the BRD classifier work into a more reproducible format.

While developing this pipeline, to get reproducibility (i.e. exact values that I obtained while prototyping), had to install dependencies via conda-forge. 

```
conda env create -n test_ploomber -f environment.yml
```

After installing the dependencies, the packaged pipeline MUST be installed with:
```
python -m pip install -e .
```

Otherwise the code will not run.

# Running pipelines

Organisation of pipelines is currently under development.

Training pipeline is in `src/brd_pcm/pipeline.yaml`, serving pipeline is in `src/brd_pcm/pipeline.serve.yaml`.
There is an additional pipeline for running extra test sets (only relevant during development) called `pipeline.extra_test.yaml`.

All the pipelines share the `pipeline.preprocessing.yaml` code.

# Notes on data

During model training, the order of the input data is NOT changed during cleaning and featurization
* Changing the order (e.g. reordering alphabetically by Protein name) can and will break the code
