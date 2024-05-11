#!/bin/sh
#SBATCH -J emrPRQA
#SBATCH -o scripts/slurm_outputs/run.out
#SBATCH -p cpu-ms

sbatch ./scripts/medication/medication0.sh
sbatch ./scripts/medication/medication66.sh
sbatch ./scripts/medication/medication77.sh
sbatch ./scripts/medication/medication89.sh
sbatch ./scripts/medication/medication100.sh
sbatch ./scripts/medication/medication119.sh
sbatch ./scripts/medication/medication121.sh
sbatch ./scripts/medication/medication142.sh
sbatch ./scripts/medication/medication229.sh

sbatch ./scripts/medication_bertb/medication0.sh
sbatch ./scripts/medication_bertb/medication66.sh
sbatch ./scripts/medication_bertb/medication77.sh
sbatch ./scripts/medication_bertb/medication89.sh
sbatch ./scripts/medication_bertb/medication100.sh
sbatch ./scripts/medication_bertb/medication119.sh
sbatch ./scripts/medication_bertb/medication121.sh
sbatch ./scripts/medication_bertb/medication142.sh
sbatch ./scripts/medication_bertb/medication229.sh

sbatch ./scripts/medication_uniform/medication0.sh
sbatch ./scripts/medication_uniform/medication66.sh
sbatch ./scripts/medication_uniform/medication77.sh
sbatch ./scripts/medication_uniform/medication89.sh
sbatch ./scripts/medication_uniform/medication100.sh
sbatch ./scripts/medication_uniform/medication119.sh
sbatch ./scripts/medication_uniform/medication121.sh
sbatch ./scripts/medication_uniform/medication142.sh
sbatch ./scripts/medication_uniform/medication229.sh

sbatch ./scripts/medication_uniform_bertb/medication0.sh
sbatch ./scripts/medication_uniform_bertb/medication66.sh
sbatch ./scripts/medication_uniform_bertb/medication77.sh
sbatch ./scripts/medication_uniform_bertb/medication89.sh
sbatch ./scripts/medication_uniform_bertb/medication100.sh
sbatch ./scripts/medication_uniform_bertb/medication119.sh
sbatch ./scripts/medication_uniform_bertb/medication121.sh
sbatch ./scripts/medication_uniform_bertb/medication142.sh
sbatch ./scripts/medication_uniform_bertb/medication229.sh

sbatch ./scripts/relations/relations0.sh
sbatch ./scripts/relations/relations68.sh
sbatch ./scripts/relations/relations134.sh
sbatch ./scripts/relations/relations143.sh
sbatch ./scripts/relations/relations149.sh
sbatch ./scripts/relations/relations154.sh
sbatch ./scripts/relations/relations157.sh
sbatch ./scripts/relations/relations190.sh
sbatch ./scripts/relations/relations198.sh

sbatch ./scripts/relations_bertb/relations0.sh
sbatch ./scripts/relations_bertb/relations68.sh
sbatch ./scripts/relations_bertb/relations134.sh
sbatch ./scripts/relations_bertb/relations143.sh
sbatch ./scripts/relations_bertb/relations149.sh
sbatch ./scripts/relations_bertb/relations154.sh
sbatch ./scripts/relations_bertb/relations157.sh
sbatch ./scripts/relations_bertb/relations190.sh
sbatch ./scripts/relations_bertb/relations198.sh

sbatch ./scripts/relations_uniform/relations0.sh
sbatch ./scripts/relations_uniform/relations68.sh
sbatch ./scripts/relations_uniform/relations134.sh
sbatch ./scripts/relations_uniform/relations143.sh
sbatch ./scripts/relations_uniform/relations149.sh
sbatch ./scripts/relations_uniform/relations154.sh
sbatch ./scripts/relations_uniform/relations157.sh
sbatch ./scripts/relations_uniform/relations190.sh
sbatch ./scripts/relations_uniform/relations198.sh

sbatch ./scripts/relations_uniform_bertb/relations0.sh
sbatch ./scripts/relations_uniform_bertb/relations68.sh
sbatch ./scripts/relations_uniform_bertb/relations134.sh
sbatch ./scripts/relations_uniform_bertb/relations143.sh
sbatch ./scripts/relations_uniform_bertb/relations149.sh
sbatch ./scripts/relations_uniform_bertb/relations154.sh
sbatch ./scripts/relations_uniform_bertb/relations157.sh
sbatch ./scripts/relations_uniform_bertb/relations190.sh
sbatch ./scripts/relations_uniform_bertb/relations198.sh