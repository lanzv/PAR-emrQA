#!/bin/sh
#SBATCH -J 135relations
#SBATCH -o scripts/slurm_outputs/relations135.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu3



python3 run_experiment.py --fts 135