#!/bin/sh
#SBATCH -J un190relations
#SBATCH -o scripts/relations_uniform/slurm_outputs/relations190.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu2



python3 run_experiment.py --fts 190 --dataset_title 'uniform' --target_average 3410