#!/bin/bash
#SBATCH --account=def-lantonie
#SBATCH --job-name=s_ols
#SBATCH --output=/home/hav/scratch/can_ai_dd/logs/%x_%j.out
#SBATCH --error=/home/hav/scratch/can_ai_dd/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G 
#SBATCH --mail-user=mhavaled@uoguelph.ca
#SBATCH --mail-type=ALL

module load python scipy-stack

python s_ols.py
