#!/bin/sh
#SBATCH -J 147relations
#SBATCH -o scripts/slurm_outputs/relations147.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu2



python3 run_experiment.py --fts 147