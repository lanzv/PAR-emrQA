#!/bin/sh
#SBATCH -J un0relations
#SBATCH -o scripts/relations_uniform/slurm_outputs/relations0.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=120G
#SBATCH --nodelist=dll-3gpu2



python3 run_experiment.py --fts 0 --dataset_title 'uniform' --target_average 254