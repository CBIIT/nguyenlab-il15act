import cupy as cp
import cupyx
import scanpy as sc
import spatialleiden as sl
import squidpy as sq
import numpy as np
from cupyx.scipy.sparse import csr_matrix
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import random
import pandas as pd
import argparse

def run(infile,outfile):
    pathout = '/data/kanferg/Sptial_Omics/projects/NguyenLab/spatialomicstoolkit/out_1'
    #andata = sc.read_h5ad(os.path.join(pathout, "adata_ctrl_2_logNorm_hvg_unintegrated.h5ad"))
    andata = sc.read_h5ad(os.path.join(pathout, infile))
    andata.obsp['connectivities'] = andata.obsp['nontumor_connectivities']
    andata.obsp['distances'] = andata.obsp['nontumor_distances']
    sq.gr.spatial_neighbors(andata, coord_type="generic")
    pathout_spatlied = "/data/kanferg/Sptial_Omics/projects/NguyenLab/spatialomicstoolkit/out_1"
    seed = 42
    sl.spatialleiden(andata, layer_ratio=1.5, directed=(False, True), seed=seed)
    andata_save = andata.copy()
    #andata_save.write_h5ad(os.path.join(pathout, "adata_ctrl_2_logNorm_hvg_unintegrated.h5ad"))
    andata_save.write_h5ad(os.path.join(pathout_spatlied, outfile))

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='calling saptiallieden')
    # Add an argument for the option selection
    parser.add_argument('--infile', type=str,required=True)
    parser.add_argument('--outfile', type=str,required=True)
    args = parser.parse_args()

    # Call the run function with the selected option and debug mode
    run(args.infile,args.outfile)

if __name__ == '__main__':
    main()