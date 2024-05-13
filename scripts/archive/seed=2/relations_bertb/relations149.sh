#!/bin/sh
#SBATCH -J bb149relations
#SBATCH -o scripts/relations_bertb/slurm_outputs/relations149.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu3



python3 run_experiment.py --model_name 'BERTbase' --model_path '../models/bert-base-cased' --fts 149