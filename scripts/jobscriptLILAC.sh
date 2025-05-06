#!/bin/env bash

#SBATCH -A NAISS2025-22-353     # project ID found via "projinfo"
#SBATCH -p alvis                # what partition to use (usually not necessary)
#SBATCH -t 0-24:00:00           # how long time it will take to run
#SBATCH --gpus-per-node=T4:1    # choosing no. GPUs and their type
#SBATCH -J ddpm         # the jobname (not necessary)

# load modules
module load virtualenv/20.26.2-GCCcore-13.3.0
module load matplotlib/3.9.2-gfbf-2024a
module load SciPy-bundle/2024.05-gfbf-2024a

# install dependencies in virtualenv
virtualenv --system-site-packages my_venv
source my_venv/bin/activate

# execute 
python run.py