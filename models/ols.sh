#!/bin/bash
#SBATCH --account=def-lantonie
#SBATCH --job-name=ols
#SBATCH --output=/home/hav/scratch/sunshine/logs/%x_%j.out
#SBATCH --error=/home/hav/scratch/sunshine/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=22
#SBATCH --mem-per-cpu=32G 
#SBATCH --mail-user=mhavaled@uoguelph.ca
#SBATCH --mail-type=ALL

module load python scipy-stack

virtualenv --no-download $SLURM_TMPDIR/env

source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index statsmodels Jinja2

python models/ols.py