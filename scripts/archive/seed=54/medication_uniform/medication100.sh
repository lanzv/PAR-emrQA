#!/bin/sh
#SBATCH -J un100medication
#SBATCH -o scripts/medication_uniform/slurm_outputs/100medication.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu3


python3 run_experiment.py --dataset_title 'uniform' --target_average 1546 --train_path './data/medication-train.json' --dev_path './data/medication-dev.json' --test_path './data/medication-test.json' --fts 100