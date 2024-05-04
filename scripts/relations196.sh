#!/bin/sh
#SBATCH -J 196relations
#SBATCH -o scripts/slurm_outputs/relations196.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu1



python3 run_experiment.py --fts 196