#!/bin/bash
#SBATCH --account=def-lantonie
#SBATCH --job-name=dd
#SBATCH --output=/home/hav/scratch/sunshine/logs/%x_%j.out
#SBATCH --error=/home/hav/scratch/sunshine/logs/%x_%j.err
#SBATCH --time=0:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G 
#SBATCH --mail-user=mhavaled@uoguelph.ca
#SBATCH --mail-type=ALL

module load python scipy-stack

virtualenv --no-download $SLURM_TMPDIR/env

source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index doubleml

python models/dd.py