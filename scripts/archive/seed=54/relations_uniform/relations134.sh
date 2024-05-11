#!/bin/sh
#SBATCH -J un134relations
#SBATCH -o scripts/relations_uniform/slurm_outputs/relations134.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu4



python3 run_experiment.py --fts 134 --dataset_title 'uniform' --target_average 1065