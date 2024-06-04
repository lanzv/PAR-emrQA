#!/bin/sh
#SBATCH -J llmrun
#SBATCH -o scripts/slurm_outputs/run_llm.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=90G
#SBATCH --nodelist=dll-3gpu1

python3 run_experiment.py --model_name 'BERTbase' --model_path '../models/bert-base-cased' --train_path './data/medication-train.json' --dev_path './data/medication-dev.json' --test_path './data/medication-test.json' --dataset_title 'medication' --fts 0