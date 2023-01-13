import click
import numpy as np
import anndata
import time
import scanpy as sc

import os
import sys

sys.path.append("../../")
from paths import DATA_DIR

DATA_DIR = DATA_DIR / "runtime_hca"
OUTPUT_DIR = DATA_DIR / "runtime_hca" / "output"


def run_sift(adata, metric="rbf", save=True):
    
    import pykeops
    # Clean up the already compiled files
    pykeops.clean_pykeops()
    
    module_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(".."))))

    sys.path.append(module_path)
    sys.path.append(os.path.join(module_path, "sift-sc"))

    import sift
    covariates = ["percent_mito", 
                  "percent_ribo",
                  "rand_noise_0",
                  "rand_noise_1",
                  "rand_noise_2",
                  "rand_noise_3",
                  "rand_noise_4",
                  "rand_noise_5",
                  "rand_noise_6",
                  "rand_noise_7"]
    
    adata.obsm["covariates"] = adata.obs[covariates].values
    
    start = time.perf_counter()
    sift.sifter(adata=adata,
                kernel_key="covariates",
                metric=metric,
                embedding_key="X",
                pseudocount=False,
                copy=False)
    compute_time = time.perf_counter() - start
    

    time_fn = f"{adata.n_obs}_{adata.n_vars}_time_sift_{metric}" 
    
    if save:
        np.save(OUTPUT_DIR + time_fn,compute_time)
        print(f"saving time of {compute_time} to {time_fn}")
    else:
        return compute_time
    

def train_scvi(adata):
    import scvi
    # bbknn_covs = ["cell_source", "donor"]
    cont_nuisance_cov = ["percent_mito", 
                         "percent_ribo",
                         "rand_noise_0",
                         "rand_noise_1",
                         "rand_noise_2",
                         "rand_noise_3",
                         "rand_noise_4",
                         "rand_noise_6",
                         "rand_noise_7"]
    
    start = time.perf_counter()
    scvi.model.SCVI.setup_anndata(
                adata,
                layer="counts",
                continuous_covariate_keys=cont_nuisance_cov                
            )
            
    m = scvi.model.SCVI(adata)
    
    if 0.1 * adata.n_obs < 20000:
        train_size = 0.9
    else:
        train_size = 1-(20000/adata.n_obs)
    print(train_size)
    
    
    m.train(early_stopping=True,
            train_size=train_size,
            early_stopping_patience=45,
            max_epochs=10000, 
            batch_size=1024, 
            limit_train_batches=20,
           )
    compute_time = time.perf_counter() - start

    
    time_fn = f"{adata.n_obs}_{adata.n_vars}_time_scvi"
    print(f"saving time of {compute_time} to {time_fn}")
    np.save(OUTPUT_DIR + time_fn, compute_time)

def run_regress(adata, save=True,):
    covariates = ["percent_mito", 
                  "percent_ribo",
                  "rand_noise_0",
                  "rand_noise_1",
                  "rand_noise_2",
                  "rand_noise_3",
                  "rand_noise_4",
                  "rand_noise_5",
                  "rand_noise_6",
                  "rand_noise_7"]
    
    start = time.perf_counter()
    sc.pp.regress_out(adata, covariates)
    compute_time = time.perf_counter() - start
    
    time_fn = f"{adata.n_obs}_{adata.n_vars}_time_regress" 
    
    if save:
        np.save(OUTPUT_DIR + time_fn, compute_time)
        print(f"saving time of {compute_time} to {time_fn}")
    else:
        return compute_time
    
    
@click.command()
@click.option("--adata")
@click.option("--method")
def run_methods(adata, method):
    
    adata = anndata.read(DATA_DIR + adata)
    
    if method == "sift-knn":
        run_sift(adata, metric="knn")
    elif method == "sift-rbf":
        run_sift(adata, metric="rbf")
    elif method == "scvi":
        train_scvi(adata)
    elif method == "regress":
        run_regress(adata)
        
if __name__ == "__main__":
    run_methods()