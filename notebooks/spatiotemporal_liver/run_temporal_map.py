from subprocess import Popen
import os
import sys

module_path = os.path.dirname(os.path.dirname(os.path.abspath(os.path.join('..'))))
sys.path.append(module_path)

DATA_DIR = "/cs/labs/mornitzan/zoe.piran/research/projects/SiFT_analysis/spatiotemporal_liver/output"
fnames  = {
    'original': '/cs/labs/mornitzan/zoe.piran/research/projects/SiFT_analysis/spatiotemporal_liver/data/processed/adata_ot.h5ad',
    'sift_knn': '/cs/labs/mornitzan/zoe.piran/research/projects/SiFT_analysis/spatiotemporal_liver/data/processed/adata_sift_spatial_genes_knn.h5ad',
    'sift_ot_layers': '/cs/labs/mornitzan/zoe.piran/research/projects/SiFT_analysis/spatiotemporal_liver/data/processed/adata_sift_ot_layers.h5ad'}
reps = 10

for r in range(reps):
    for key, fname in fnames.items():
        if key != "original":
            cmdline1 = ['sbatch', '--gres=gpu:1', '--mem=200gb', '-c1', '--time=04:00:00',
                        f'--output={DATA_DIR}/logs/{key}-X_scvi_sift_psd-{r}.log',
                        f'--job-name=map-{key}-X_scvi_sift-{r}',
                        'temporal_map.sh', str(fname), "X_scvi_sift", f'{DATA_DIR}/res_rng/{key}_X_scvi_sift_psd-{r}', "X_sift"
                        ]
            print(' '.join(cmdline1))
            Popen(cmdline1)
                
        else:
            cmdline0 = ['sbatch', '--gres=gpu:1', '--mem=200gb', '-c1', '--time=04:00:00',
                        f'--output={DATA_DIR}/logs/{key}-X_scvi-{r}.log',
                        f'--job-name=map-{key}-X_scvi-{r}',
                        'temporal_map.sh', str(fname), "X_scvi", f'{DATA_DIR}/res_rng/{key}_X_scvi-{r}', "X"
                        ]
            print(' '.join(cmdline0))
            Popen(cmdline0)
                
                
