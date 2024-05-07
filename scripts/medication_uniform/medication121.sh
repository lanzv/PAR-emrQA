#!/bin/sh
#SBATCH -J un121medication
#SBATCH -o scripts/medication_uniform/slurm_outputs/121medication.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu2


python3 run_experiment.py --dataset_title 'uniform' --target_average 2212 --train_path './data/medication-train.json' --dev_path './data/medication-dev.json' --test_path './data/medication-test.json' --fts 121