# mamba activate pymc_latest
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
import seaborn as sns
import os
import gzip
import numpy as np
import statsmodels.api as sm
from scipy.special import softmax
import random
import re
import arviz as az
import xarray as xr
import pymc as pm
import math
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

import pickle

import anndata as ad

class posterior_analysis_dge_poisson:
    def __init__(self, path_data, trace_name, data_name):
        self.path_data = path_data
        self.trace_name = trace_name
        self.data_name = data_name
        self.df_long, self.trace = self.load_data()
        
        
    def load_data(self):
        df_long = pd.read_csv(os.path.join(self.path_data,self.data_name))
        with open(os.path.join(self.path_data, self.trace_name), 'rb') as buff:
            trace = pickle.load(buff)
        return df_long, trace
    
    
    
class analysis_toxicology(posterior_analysis_dge_poisson):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_tox_macrophage = self.campute_posterior_tables()
    
    def campute_posterior_tables(self):
        """
        Returns a table with columns:
        gene_name | FC (median) | hdi_lower | hdi_upper | pv (two-sided)
        """
        condtion = ['IL-15', 'Both']
        df_long = self.df_long
        trace = self.trace
        genes_order = pd.unique(df_long['gene_name']).tolist()
        
        ind_genes, genes_levels   = df_long['gene_name'].factorize()       # shape (N,)
        ind_cond,    cond_levels      = df_long['condition'].factorize()     # shape (N,)
        ind_tissue,     tissue_levels   = df_long['tissue'].factorize()        # shape (N,)
        
        map_gene = {indg:g for indg,g in zip(trace['trace'].posterior['beta'].beta_dim_0.values,genes_levels)}
        map_cond = {indc:c for indc,c in zip(trace['trace'].posterior['beta'].beta_dim_1.values,cond_levels)}
        map_tissue = {indt:t for indt,t in zip(trace['trace'].posterior['beta'].beta_dim_2.values,tissue_levels)}
        
        beta_samples = trace['trace'].posterior['beta'].mean(dim=('chain'))
        beta_long = beta_samples.to_dataframe().reset_index()
        beta_long.rename(columns = {'beta_dim_0':'gene_name'},inplace=True)
        beta_long.rename(columns = {'beta_dim_1':'condition'},inplace=True)
        beta_long.rename(columns = {'beta_dim_2':'tissue'},inplace=True)
        beta_long['gene_name'] = beta_long['gene_name'].map(map_gene)
        beta_long['condition'] = beta_long['condition'].map(map_cond)
        beta_long['tissue'] = beta_long['tissue'].map(map_tissue)
        
        beta_long = beta_long.loc[beta_long['condition'].isin(condtion),:]
                            
        table_list = []
        for tis in ['Liver', 'Kidney', 'Lung', 'Eye', 'Spleen']:
            df = self.heatmapTables(df = beta_long,genes_order=genes_order,tissue=tis)
            table_list.append(df)
            del(df)   
        table_tox_macrophage = pd.concat(table_list)
        return table_tox_macrophage        
    
    
    
    def compute_p_two(arr):
        """
        Two-sided tail probability analogue:
        p_two = 2 * min(P(arr > 0), P(arr < 0)).
        """
        arr = np.asarray(arr).ravel()
        p_pos = np.mean(arr > 0)
        p_two = 2 * min(p_pos, 1 - p_pos)
        return float(p_two)
    
    @staticmethod
    def heatmapTables(df, genes_order, tissue, hdi_prob=0.95):
        """
        Returns a table with columns:
        gene_name | FC (median) | hdi_lower | hdi_upper | pv (two-sided)

        Assumes df has columns: tissue, gene_name, condition, beta
        where beta contains samples/draws for that condition.
        """
        def compute_p_two(arr):
            """
            Two-sided tail probability analogue:
            p_two = 2 * min(P(arr > 0), P(arr < 0)).
            """
            arr = np.asarray(arr).ravel()
            p_pos = np.mean(arr > 0)
            p_two = 2 * min(p_pos, 1 - p_pos)
            return float(p_two)
        
        df_curr = df.loc[df["tissue"] == tissue].copy()

        rows = []
        for gene in genes_order:
            df_g = df_curr.loc[df_curr["gene_name"] == gene]

            il15 = df_g.loc[df_g["condition"] == "IL-15", "beta"].values
            both = df_g.loc[df_g["condition"] == "Both",  "beta"].values

            # Optional safety check (skip if missing)
            if len(il15) == 0 or len(both) == 0:
                rows.append({
                    "tissue": tissue,
                    "gene_name": gene,
                    "FC": np.nan,
                    "hdi_lower": np.nan,
                    "hdi_upper": np.nan,
                    "pv": np.nan,
                    "n_il15": len(il15),
                    "n_both": len(both),
                })
                continue

            # Define FC direction
            fc = both - il15
            fc = np.asarray(fc).ravel()
            hdi = az.hdi(fc, hdi_prob=hdi_prob)
            # ensure scalars (az.hdi returns array-like)
            hdi_lo = float(np.asarray(hdi)[0])
            hdi_hi = float(np.asarray(hdi)[1])

            pv = compute_p_two(fc)

            rows.append({
                "tissue": tissue,
                "gene_name": gene,
                "FC": float(np.median(fc)),
                "hdi_lower": hdi_lo,
                "hdi_upper": hdi_hi,
                "pv": pv,
                "n_il15": int(len(il15)),
                "n_both": int(len(both)),
            })

        out = pd.DataFrame(rows)
        return out
    
    
class analysis_skin(posterior_analysis_dge_poisson):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_skin = self.campute_posterior_tables()
    
    @staticmethod
    def heatmapTables(df, genes_order, condition, baseline, hdi_prob=0.95):
        """
        Returns a table with columns:
        gene_name | FC (median) | hdi_lower | hdi_upper | pv (two-sided)

        Assumes df has columns: gene_name, condition, beta
        where beta contains samples/draws for that condition.
        
        Args:
            baseline: The reference condition (e.g., 'TCR', 'IL-15')
            condition: The target condition to compare against baseline
        """
        def compute_p_two(arr):
            """
            Two-sided tail probability analogue:
            p_two = 2 * min(P(arr > 0), P(arr < 0)).
            """
            arr = np.asarray(arr).ravel()
            p_pos = np.mean(arr > 0)
            p_two = 2 * min(p_pos, 1 - p_pos)
            return float(p_two)
        
        rows = []
        for gene in genes_order:
            df_g = df.loc[df["gene_name"] == gene].copy()

            baseline_vals = df_g.loc[df_g["condition"] == baseline, "beta"].values
            target_vals = df_g.loc[df_g["condition"] == condition, "beta"].values
            
            # Optional safety check (skip if missing)
            if len(target_vals) == 0 or len(baseline_vals) == 0:
                rows.append({
                    "comparison": f'{condition}_vs_{baseline}',
                    "gene_name": gene,
                    "FC": np.nan,
                    "hdi_lower": np.nan,
                    "hdi_upper": np.nan,
                    "pv": np.nan,
                    "n_target": len(target_vals),
                    "n_baseline": len(baseline_vals),
                })
                continue

            # Define FC direction: target - baseline
            fc = target_vals - baseline_vals
            fc = np.asarray(fc).ravel()
            hdi = az.hdi(fc, hdi_prob=hdi_prob)
            # ensure scalars (az.hdi returns array-like)
            hdi_lo = float(np.asarray(hdi)[0])
            hdi_hi = float(np.asarray(hdi)[1])

            pv = compute_p_two(fc)

            rows.append({
                "comparison": f'{condition}_vs_{baseline}',
                "gene_name": gene,
                "FC": float(np.median(fc)),
                "hdi_lower": hdi_lo,
                "hdi_upper": hdi_hi,
                "pv": pv,
                "n_target": len(target_vals),
                "n_baseline": len(baseline_vals),
            })

        out = pd.DataFrame(rows)
        return out
    
    def campute_posterior_tables(self):
        """
        Returns a table with columns:
        gene_name | FC (median) | hdi_lower | hdi_upper | pv (two-sided)
        
        Compares multiple conditions against different baselines:
        - IL-15, IL-21, Both vs TCR
        - IL-21, Both vs IL-15
        """
        df_long = self.df_long
        trace = self.trace['trace']
        genes_order = pd.unique(df_long['gene_name']).tolist()
        
        ind_genes, genes_levels   = df_long['gene_name'].factorize()
        ind_cond, cond_levels     = df_long['condition'].factorize()
        map_gene = {indg:g for indg,g in zip(trace.posterior['beta'].beta_dim_0.values, genes_levels)}
        map_cond = {indc:c for indc,c in zip(trace.posterior['beta'].beta_dim_1.values, cond_levels)}
        
        beta_samples = trace.posterior['beta'].mean(dim=('chain'))
        beta_long = beta_samples.to_dataframe().reset_index()
        beta_long.rename(columns={'beta_dim_0':'gene_name', 'beta_dim_1':'condition'}, inplace=True)
        beta_long['gene_name'] = beta_long['gene_name'].map(map_gene)
        beta_long['condition'] = beta_long['condition'].map(map_cond)
        
        cond_list = []
        
        # Comparisons vs TCR (baseline)
        for cond in ['IL-15', 'IL-21', 'Both']:
            df = self.heatmapTables(df=beta_long, genes_order=genes_order, 
                                   condition=cond, baseline='TCR')
            cond_list.append(df)
        
        # Comparisons vs IL-15 (baseline)
        for cond in ['IL-21', 'Both']:
            df = self.heatmapTables(df=beta_long, genes_order=genes_order, 
                                   condition=cond, baseline='IL-15')
            cond_list.append(df)
        
        table_skin = pd.concat(cond_list, ignore_index=True)
        return table_skin
    
    def save_files(self, outh_path,file_name, celltype,day):
        self.table_skin['cell_type'] = celltype
        self.table_skin['day'] = day
        self.table_skin.to_csv(f'{outh_path}/{file_name}', index=False)