#!/bin/sh
#SBATCH -J unbb134relations
#SBATCH -o scripts/relations_uniform_bertb/slurm_outputs/relations134.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu4



python3 run_experiment.py --model_name 'BERTbase' --model_path '../models/bert-base-cased' --fts 134 --dataset_title 'uniform' --target_average 1065