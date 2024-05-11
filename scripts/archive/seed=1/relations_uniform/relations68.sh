#!/bin/sh
#SBATCH -J un68relations
#SBATCH -o scripts/relations_uniform/slurm_outputs/relations68.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu4



python3 run_experiment.py --fts 68 --dataset_title 'uniform' --target_average 667