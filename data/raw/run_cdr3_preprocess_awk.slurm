#!/bin/bash
#SBATCH --job-name=oas_cdr3
#SBATCH --account=vusr120775
#SBATCH --partition=gpu_a100
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=oas_cdr3.out
#SBATCH --error=oas_cdr3.err

echo "Start: $(date)"
echo "Node: $(hostname)"
echo "TMPDIR=$TMPDIR"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

bash process_oas_covid_human_heavy_cdr3_awk.sh

echo "End: $(date)"
