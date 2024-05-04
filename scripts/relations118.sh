#!/bin/sh
#SBATCH -J 118relations
#SBATCH -o scripts/slurm_outputs/relations118.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu4



python3 run_experiment.py --fts 118