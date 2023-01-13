import sys

import os
import numpy as np

import scanpy as sc
import pandas as pd

import argparse

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
sys.path.append("/cs/labs/mornitzan/zoe.piran/research/projects/SiFT_analysis/spatiotemporal_liver")


module_path = os.path.dirname(os.path.abspath(os.path.join('..')))
sys.path.append(module_path)

from mapping import MappingProblem, get_mapping


def temporal_map(args):
    fused_penalty = [0.001, 0.005, 0.01, 0.1, 0.5]
    epsilons = [1e-4, 1e-3, 1e-2, 1e-1]
    markers = [0, 5, 10]
    DATA_DIR = "/cs/labs/mornitzan/zoe.piran/research/projects/SiFT_analysis/spatiotemporal_liver/data/"

    adata_spatial = sc.read(DATA_DIR+"processed/adata_ZT.h5ad")
    temporal_genes_ref = [
        "arntl", "clock", "npas2",
        "nr1d1", "nr1d2", "per1",
        "per2", "cry1", "cry2",
        "dbp", "tef", "hlf",
        "elovl3", "rora", "rorc"]


    ref_temporal_genes = adata_spatial[:, adata_spatial.var["type"].isin([3])].var_names[np.argsort(
        adata_spatial[:, adata_spatial.var["type"].isin([3])].var["gene_confidence"])][-20:]
    
    df_res = pd.DataFrame(columns=['type',
                                   'fused_penalty',
                                   'epsilon',
                                   'markers',
                                   'num_markers',
                                   'pears_mean',
                                   'converged',
                                   'out',
                                   ])

    dataset = {
        "adata_spatial": DATA_DIR+"processed/adata_ZT.h5ad",
        "adata_sc": args.adata,
        "sc_key": args.sc_key,
        "sp_key": "spatial",
        "sc_joint_key": args.sc_joint_key,
        "reference_names": ref_temporal_genes,
    }

    solver = {"rank": None
              }

    training = {}

    for marker in markers:
        marker_names = np.random.default_rng().choice(temporal_genes_ref, size=int(marker), replace=False)
        dataset["marker_names"] = marker_names
        print(f"evaluating markers: {marker_names}")

        for epsilon in epsilons:
            print(f"evaluating epsilon {epsilon}")
            solver["epsilon"] = epsilon
            if marker == 0:
                fp = 0
                dataset["marker_names"] = None
                print(f"evaluating alpha {fp}")
                solver["fused_penalty"] = fp
                ## Solve

                mapping = get_mapping()
                mapping.init_model(solver)
                mapping.init_dataset(dataset)
                res_, out = mapping.train(training)

                dict_ = {}
                dict_['epsilon'] = mapping.epsilon
                dict_['fused_penalty'] = mapping.fused_penalty
                dict_['markers'] = mapping.adata_spatial.var_names[mapping.adata_spatial.var["marker"]]
                dict_['num_markers'] = marker
                dict_['pears_mean'] = res_['pears_mean']
                dict_['pears_std'] = res_['pears_std']
                dict_['accuracy'] = res_['accuracy']
                dict_['converged'] = res_['converged']
                print(dict_)
                dict_['out'] = out

                df_res = df_res.append(dict_, ignore_index=True)

            else:
                for fp in fused_penalty:
                    print(f"evaluating fused_penalty {fp}")
                    solver["fused_penalty"] = fp

                    ## Solve
                    mapping = get_mapping()
                    mapping.init_model(solver)
                    mapping.init_dataset(dataset)
                    res_, out = mapping.train(training)

                    dict_ = {}
                    dict_['epsilon'] = mapping.epsilon
                    dict_['fused_penalty'] = mapping.fused_penalty
                    dict_['markers'] = mapping.adata_spatial.var_names[mapping.adata_spatial.var["marker"]]
                    dict_['num_markers'] = marker
                    dict_['pears_mean'] = res_['pears_mean']
                    dict_['pears_std'] = res_['pears_std']
                    dict_['accuracy'] = res_['accuracy']
                    dict_['converged'] = res_['converged']
                    
                    print(dict_)
                    dict_['out'] = out

                    df_res = df_res.append(dict_, ignore_index=True)
    df_res.to_pickle(args.out)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adata',
                        type=str,
                        required=True)
    parser.add_argument('--sc_key',
                        type=str,
                        required=False,
                        default="X_scvi")
    parser.add_argument('--out',
                        type=str,
                        required=True,)
    parser.add_argument('--sc_joint_key',
                        type=str,
                        required=False,
                        default="X")
    args = parser.parse_args()

    temporal_map(args)
