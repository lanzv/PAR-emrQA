#!/bin/sh
#SBATCH -J 121medication
#SBATCH -o scripts/medication/slurm_outputs/121medication.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu2


python3 run_experiment.py --train_path './data/medication-train.json' --dev_path './data/medication-dev.json' --test_path './data/medication-test.json' --dataset_title 'medication' --fts 121