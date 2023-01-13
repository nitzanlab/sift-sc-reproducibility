#!/bin/bash

PROJECT_DIR="/cs/labs/mornitzan/zoe.piran/research/projects/SiFT_analysis/spatiotemporal_liver"

adata=$1
sc_key=$2
out=$3
sc_joint_key=$4

module load python/3.9
module load cuda
module load tensorflow
source /cs/labs/mornitzan/zoe.piran/venvsc/bin/activate

python3 ${PROJECT_DIR}/temporal_map.py --adata ${adata} --sc_key ${sc_key} --out ${out} --sc_joint_key ${sc_joint_key}