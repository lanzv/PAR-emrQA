#!/bin/sh
#SBATCH -J 154relations
#SBATCH -o scripts/relations/slurm_outputs/relations154.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu2



python3 run_experiment.py --fts 154