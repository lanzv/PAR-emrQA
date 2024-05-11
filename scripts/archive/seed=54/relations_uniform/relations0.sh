#!/bin/sh
#SBATCH -J un0relations
#SBATCH -o scripts/relations_uniform/slurm_outputs/relations0.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=90G
#SBATCH --nodelist=tdll-3gpu3



python3 run_experiment.py --fts 0 --dataset_title 'uniform' --target_average 254