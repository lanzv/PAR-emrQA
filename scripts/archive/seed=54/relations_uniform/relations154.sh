#!/bin/sh
#SBATCH -J un154relations
#SBATCH -o scripts/relations_uniform/slurm_outputs/relations154.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu3



python3 run_experiment.py --fts 154 --dataset_title 'uniform' --target_average 2374