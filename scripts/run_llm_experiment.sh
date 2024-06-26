#!/bin/sh
#SBATCH -J llmrun
#SBATCH -o scripts/slurm_outputs/run_llm.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=90G
#SBATCH --nodelist=dll-3gpu1

python3 run_experiment.py --model_name 'BioMistral' --model_path '../models/BioMistral-7B' --train_path './data/relations-train.json' --dev_path './data/relations-dev.json' --test_path './data/relations-test.json' --dataset_title 'relations' --fts 0