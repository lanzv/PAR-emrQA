#!/bin/sh
#SBATCH -J unbb157relations
#SBATCH -o scripts/relations_uniform_bertb/slurm_outputs/relations157.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu2



python3 run_experiment.py --model_name 'BERTbase' --model_path '../models/bert-base-cased' --fts 157 --dataset_title 'uniform' --target_average 3041