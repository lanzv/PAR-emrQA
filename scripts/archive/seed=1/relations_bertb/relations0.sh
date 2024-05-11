#!/bin/sh
#SBATCH -J bb0relations
#SBATCH -o scripts/relations_bertb/slurm_outputs/relations0.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=120G
#SBATCH --nodelist=dll-3gpu2



python3 run_experiment.py --model_name 'BERTbase' --model_path '../models/bert-base-cased' --fts 0