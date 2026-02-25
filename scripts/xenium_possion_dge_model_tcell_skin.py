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
import argparse
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
# import jax

def slurm_progress(trace, draw):
    if getattr(draw, "tuning", False):
        return
    if len(trace) % 200 == 0:
        print(f"[chain {getattr(draw,'chain','?')}] draws={len(trace)}", flush=True)


def run(infile,outfile,day,celltype):
    pathmodel = '/data/HiTIF/data/spatialomics/melanoma/data/models/possion_dge'
    df_long_all = pd.read_csv(os.path.join(pathmodel,infile))
    df_long = df_long_all.query('annotation == @celltype and day == @day').reset_index(drop=True)
    '''
    Incase the modeling is to long: 
    test.groupby(['condition', 'batch', 'day', 'annotation', 'gene_name','size'],as_index=False)['expr'].mean()
    create a mean data frame pair batch
    '''
    
    ind_genes, genes_levels   = df_long['gene_name'].factorize()       # shape (N,)
    ind_cond,    cond_levels      = df_long['condition'].factorize()     # shape (N,)
    ind_batch,     batch_levels   = df_long['batch'].factorize()        # shape (N,)

    y = df_long['expr'].values                                  # shape (N,)
    N = np.log(df_long['size'].values)

    # # Sizes
    G = len(genes_levels)   ## clusters
    C = len(cond_levels)  ## conditions
    B = len(batch_levels)       ## days


    coords = {
        "obs_id": np.arange(y.shape[0]),
        "Genes":  genes_levels,  
        "Condition": cond_levels,          # length C
        "batch": batch_levels
    }

    with pm.Model(coords=coords) as model:
        g = pm.Data("g", ind_genes, dims=("obs_id",))
        c = pm.Data("c", ind_cond, dims=("obs_id",))
        b = pm.Data("b", ind_batch, dims=("obs_id",))
        offset = pm.Data("logN", np.log(df_long['size'].to_numpy()), dims=("obs_id",))

        beta_bar = pm.Normal("beta_bar", 0,0.5)
        sigma_bar = pm.HalfNormal("sigma_bar",1)
        z_beta = pm.Normal("z_beta", 0, 1, dims=("Genes","Condition"))
        beta = pm.Deterministic("beta", beta_bar + z_beta*sigma_bar)
        
        sigma_lam = pm.HalfNormal("sigma_lam", 0.5)
        z_lam = pm.Normal("z_lam", 0, 0.5, dims=("Genes"))
        lam = pm.Deterministic("lam", z_lam * sigma_lam)
        
        sigma_delta = pm.HalfNormal("sigma_delta", 0.5)
        z_delta = pm.Normal("z_delta", 0, 1, dims=("Condition"))
        delta = pm.Deterministic("delta", z_delta * sigma_delta)

        sigma_rho = pm.HalfNormal("sigma_rho", 0.5)
        z_rho = pm.Normal("z_rho", 0, 1, dims=("batch"))
        rho = pm.Deterministic("rho", z_rho * sigma_rho)
        
        # theta_pre = pm.Deterministic('theta_pre',lam[g,None,None]+ delta[None,c,None] + tau[None,None,s] + beta[g,c,s]*N[None,:,None])
        #theta_pre = pm.Deterministic('theta_pre',lam[g,None,None]+ delta[None,c,None] + tau[None,None,s] + beta[g,c,s]*N[None,:,None])
        eta = pm.Deterministic("eta", offset + lam[g] + delta[c] + beta[g, c]) + rho[b]
        theta = pm.math.exp(eta)

        y_obs = pm.Poisson("y_obs", mu=theta, observed=y)
        trace = pm.sample(1000, tune=1000, target_accept=0.999, chains=4,
    cores=4, progressbar=False, callback=slurm_progress, random_seed=RANDOM_SEED)

    import pickle
    with open(os.path.join(pathmodel, outfile), 'wb') as buff:
        pickle.dump({'trace':trace,'celltype':celltype,'day':day}, buff)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='calling dge possion model')
    # Add an argument for the option selection
    parser.add_argument('--infile', type=str,required=True)
    parser.add_argument('--outfile', type=str,required=True)
    parser.add_argument('--day', type=int, required=True)
    parser.add_argument('--celltype', type=str, required=True)  
    args = parser.parse_args()

    # Call the run function with the selected option and debug mode
    run(args.infile,args.outfile,args.day,args.celltype)

if __name__ == '__main__':
    main()