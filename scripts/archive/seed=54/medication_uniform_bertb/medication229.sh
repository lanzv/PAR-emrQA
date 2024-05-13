#!/bin/sh
#SBATCH -J unbb229medication
#SBATCH -o scripts/medication_uniform_bertb/slurm_outputs/229medication.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu2


python3 run_experiment.py --model_name 'BERTbase' --model_path '../models/bert-base-cased' --dataset_title 'uniform' --target_average 6672 --train_path './data/medication-train.json' --dev_path './data/medication-dev.json' --test_path './data/medication-test.json' --fts 229