#!/bin/sh
#SBATCH -J un0medication
#SBATCH -o scripts/medication_uniform/slurm_outputs/0medication.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=90G
#SBATCH --nodelist=dll-3gpu5


python3 run_experiment.py --dataset_title 'uniform' --target_average 339 --train_path './data/medication-train.json' --dev_path './data/medication-dev.json' --test_path './data/medication-test.json' --fts 0