# Notebooks ↔ dataset files (Zenodo)

> Relative paths below assume the Zenodo bundle is unpacked into `data_external/nguyenlab-il15act-dataset-v1/`.
> Both Zenodo uploads contain the large objects needed to run the pipeline end-to-end:
> https://zenodo.org/uploads/18838969 and https://zenodo.org/uploads/18842625.

| Notebook | Primary inputs from Zenodo bundle (relative paths) |
|---|---|
| `nb_01_qc_cell_transcript_filtering.ipynb` | `01_processed_anndata/mel_cytokines_andata.h5ad` • `00_metadata/` (metadata notes) |
| `nb_02_compostion_analysis.ipynb` | `01_processed_anndata/mel_cytokines_andata.h5ad` |
| `nb_03_tumor_regions_compostion_analysis.ipynb` | `02_tables_inputs/sccoda_tumor_regions_input.csv` |
| `nb_04_ligand_receptor_analysis.ipynb` | `03_lr_outputs/all_lr_interactions_scores.csv` |
| `nb_05_Collagen_distance_bayes_model.ipynb` | `02_tables_inputs/colga_distance_input_table.csv` |
| `nb_06_xenium_codex_alignment.ipynb` | `04_codex_slide1/` (Xenium/CODEX exports; TIFF/QPTIFF-derived tables) • `05_spatialdata_zarr/` (raw + corrected `.zarr` objects) |
| `nb_07_toxicology_compostion_analysis.ipynb` | `01_processed_anndata/tox_cytokines_andata.h5ad` |
| `nb_08_tcell_signature_gene_analysis.ipynb` | `02_tables_inputs/skin_tcell_long_counts.csv` • `06_models/possion_dge/trace_cd8subsets_day4.pkl` |
