# nguyenlab-il15act



## Environments (mamba/conda)

This project uses multiple conda environments because some notebooks require incompatible package stacks
(e.g., scCODA/JAX vs PyMC vs stLearn vs spatialdata tooling). Minimal environment specs are provided in `envs/`.

### Create environments
From the repository root:

```bash
# scCODA (composition modeling)
mamba env create -f envs/env_scoda.min.yml

# PyMC (Bayesian modeling; used by some downstream notebooks/scripts)
mamba env create -f envs/env_pymc.min.yml

# stLearn-based workflows
mamba env create -f envs/env_stlearn.min.yml

# spatialdata / spatial ecosystem workflows
mamba env create -f envs/env_spatialdata.min.yml
```
## Data availability (Zenodo)

Large input files required to reproduce the notebook analyses (e.g., processed AnnData objects and other intermediate analysis inputs) are hosted on Zenodo in two uploads:

- https://zenodo.org/uploads/18838969
- https://zenodo.org/uploads/18842625

These Zenodo uploads provide the processed datasets needed to run the notebooks end-to-end, including the precomputed inputs used across the pipeline (e.g., the Xenium-derived AnnData used for QC/composition analyses, and analysis-ready tables used by downstream modeling steps such as region composition/scCODA, collagen distance modeling, and ligand–receptor analysis). The repository itself only tracks small, analysis-ready tables under `data/` (CSV inputs and summary tables), while the larger objects are downloaded from Zenodo.

After downloading, place the Zenodo files in a local data directory (recommended: `data_external/` at the repo root) and update the path variables at the top of each notebook to point to your local file locations.

