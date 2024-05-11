#!/bin/sh
#SBATCH -J llmrun
#SBATCH -o scripts/slurm_outputs/run_llm.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=90G
#SBATCH --nodelist=dll-3gpu5

python llm.py