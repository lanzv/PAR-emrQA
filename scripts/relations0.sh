#!/bin/sh
#SBATCH -J 0relations
#SBATCH -o scripts/slurm_outputs/relations0.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu5



python3 run_experiment.py --fts 0