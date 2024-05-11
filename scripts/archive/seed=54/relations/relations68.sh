#!/bin/sh
#SBATCH -J 68relations
#SBATCH -o scripts/relations/slurm_outputs/relations68.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu4



python3 run_experiment.py --fts 68