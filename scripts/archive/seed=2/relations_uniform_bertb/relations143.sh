#!/bin/sh
#SBATCH -J unbb143relations
#SBATCH -o scripts/relations_uniform_bertb/slurm_outputs/relations143.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu3



python3 run_experiment.py --model_name 'BERTbase' --model_path '../models/bert-base-cased' --fts 143 --dataset_title 'uniform' --target_average 1455