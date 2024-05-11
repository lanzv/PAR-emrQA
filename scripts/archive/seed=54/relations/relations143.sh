#!/bin/sh
#SBATCH -J 143relations
#SBATCH -o scripts/relations/slurm_outputs/relations143.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu3



python3 run_experiment.py --fts 143