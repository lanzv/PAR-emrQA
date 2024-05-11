#!/bin/sh
#SBATCH -J 190relations
#SBATCH -o scripts/relations/slurm_outputs/relations190.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu1



python3 run_experiment.py --fts 190